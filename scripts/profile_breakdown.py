"""
Profile MeZO vs GRZO wall-clock on multi-GPU.

Usage (in a 1-node 4-GPU alloc):
    torchrun --standalone --nnodes=1 --nproc-per-node=4 \
        scripts/profile_breakdown.py \
        --methods mezo,grzo,mezo_forward_only,mezo_perturb_only,mezo_noop_traversal,grzo_forward_no_flipout,grzo_sign_only \
        --warmup-steps 5 --profile-steps 20 \
        --model /path/to/Meta-Llama-3-8B \
        --batch-size 4 --seq-len 512 \
        --output profiling/breakdown

Each method gets its own JSONL per rank; rank 0 aggregates to summary.csv + report.md.

Note: we re-implement minimal MeZO/GRZO step here so attribution is clean.
For GRZO, we call optimizers/flipout.grzo_flipout_step_ddp under NVTX wrap.
"""

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import nvtx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "optimizers"))
sys.path.insert(0, str(REPO_ROOT / "MeZO" / "large_models"))

try:
    from flipout import grzo_flipout_step_ddp, grzo_flipout_step
except ImportError as e:
    print(f"WARN: couldn't import flipout: {e}", file=sys.stderr)
    grzo_flipout_step_ddp = None
    grzo_flipout_step = None


# ----------------------------------------------------------------------------
# Timer / NVTX helpers
# ----------------------------------------------------------------------------

class Timer:
    """Context manager: torch.cuda.synchronize + perf_counter + NVTX range.

    Records to a dict {range_name: [t_ms, ...]}.
    """

    def __init__(self, store, name, sync=True, nvtx_name=None):
        self.store = store
        self.name = name
        self.sync = sync
        self.nvtx_name = nvtx_name or name

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        nvtx.range_push(self.nvtx_name)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.sync:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
        nvtx.range_pop()
        self.store.setdefault(self.name, []).append(elapsed_ms)


@contextlib.contextmanager
def nvtx_range(name):
    """Cheap fine-grained NVTX (no sync)."""
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


# ----------------------------------------------------------------------------
# MeZO step (minimal, mirrors our trainer.py:1500 zo_step)
# ----------------------------------------------------------------------------

def _iter_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _gen_z_like(p, seed_state):
    """Generate a Gaussian z on the same device/dtype as p, deterministic by seed_state."""
    g = torch.Generator(device=p.device)
    g.manual_seed(seed_state)
    return torch.randn(p.data.size(), generator=g, device=p.device, dtype=p.dtype)


def _perturb_inplace(params, seed, eps, alpha):
    """In-place: p += alpha * eps * z(p, seed). Uses fresh torch.manual_seed (matches trainer.py)."""
    torch.manual_seed(seed)
    for p in params:
        z = torch.normal(mean=0.0, std=1.0, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
        p.data.add_(z, alpha=alpha * eps)


def mezo_step(model, batch, eps, lr, seed, store, sync_breakdown=True):
    """One vanilla MeZO step with per-component timing."""
    params = _iter_trainable_params(model)

    with Timer(store, "mezo_step_total", sync=True):
        # 1. perturb +z
        with Timer(store, "mezo_perturb_pos", sync=sync_breakdown, nvtx_name="mezo_perturb_pos"):
            _perturb_inplace(params, seed, eps, alpha=+1.0)

        # 2. forward +z
        with Timer(store, "mezo_forward_pos", sync=sync_breakdown):
            with torch.no_grad():
                loss_pos = model(**batch).loss.detach()

        # 3. perturb -2z
        with Timer(store, "mezo_perturb_neg", sync=sync_breakdown):
            _perturb_inplace(params, seed, eps, alpha=-2.0)

        # 4. forward -z
        with Timer(store, "mezo_forward_neg", sync=sync_breakdown):
            with torch.no_grad():
                loss_neg = model(**batch).loss.detach()

        # 5. restore (+z)
        with Timer(store, "mezo_restore", sync=sync_breakdown):
            _perturb_inplace(params, seed, eps, alpha=+1.0)

        # 6. projected grad scalar
        with Timer(store, "mezo_loss_diff", sync=sync_breakdown):
            proj_grad = (loss_pos - loss_neg) / (2.0 * eps)

        # 7. all_reduce projected_grad across ranks
        with Timer(store, "mezo_allreduce_scalar", sync=sync_breakdown):
            if dist.is_initialized():
                dist.all_reduce(proj_grad, op=dist.ReduceOp.SUM)
                proj_grad = proj_grad / dist.get_world_size()

        # 8. update p -= lr * proj_grad * z(seed)
        with Timer(store, "mezo_update", sync=sync_breakdown):
            torch.manual_seed(seed)
            for p in params:
                z = torch.normal(mean=0.0, std=1.0, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                p.data.add_(z, alpha=-lr * float(proj_grad))


def _quant_sym(x, nbits):
    """Symmetric quant+dequant (deterministic round), matches mezo_quzo path."""
    n_levels = (1 << (nbits - 1)) - 1
    abs_max = x.detach().abs().max().clamp(min=1e-12)
    scale = n_levels / abs_max
    q = (x * scale).round().clamp(-n_levels, n_levels)
    return (q / scale).to(x.dtype)


def _perturb_inplace_variant(params, seed, eps, alpha, variant, sparse_thresholds=None,
                              lozo_rank=4, quant_bits=4):
    """In-place: p += alpha*eps*z_modified. variant ∈ {vanilla,sparse,lozo,quzo}."""
    torch.manual_seed(seed)
    for p in params:
        z = torch.normal(mean=0.0, std=1.0, size=p.data.size(),
                         device=p.data.device, dtype=p.data.dtype)
        if variant == "sparse" and sparse_thresholds is not None and id(p) in sparse_thresholds:
            # Compute mask on-demand from threshold (saves 16GB cache)
            thr = sparse_thresholds[id(p)]
            mask = (p.data.detach().abs() <= thr).to(z.dtype)
            z = z * mask
        elif variant == "lozo" and p.data.dim() == 2:
            # Approximate per-param low-rank: u (dout, r) @ v (r, din) / sqrt(r)
            dout, din = p.data.size()
            g = torch.Generator(device=p.device); g.manual_seed(seed + p.data.size(0))
            u = torch.randn(dout, lozo_rank, generator=g, device=p.device, dtype=p.dtype)
            v = torch.randn(lozo_rank, din, generator=g, device=p.device, dtype=p.dtype)
            z = (u @ v) / (lozo_rank ** 0.5)
        elif variant == "quzo":
            z = _quant_sym(z, quant_bits)
        p.data.add_(z, alpha=alpha * eps)


def mezo_variant_step(model, batch, eps, lr, seed, store, variant, sync_breakdown=True,
                       sparse_ratio=0.20, lozo_rank=4, quant_bits=4):
    """Generic MeZO step with variant ∈ {vanilla, sparse, lozo, quzo}.
       Two-sided: 2 forwards, 3 perturbs, 1 update — all with variant-specific noise."""
    params = _iter_trainable_params(model)

    # Build sparse THRESHOLDS (per-param scalar) if needed — recompute mask on-demand
    # to avoid storing 16GB of masks.
    sparse_thresholds = None
    if variant == "sparse":
        if not hasattr(model, '_mezo_sparse_thr'):
            model._mezo_sparse_thr = {}
            for p in params:
                if p.data.dim() >= 2:
                    n = p.data.numel()
                    k = max(1, int(sparse_ratio * n))
                    thr = torch.kthvalue(p.data.detach().abs().view(-1), k)[0]
                    model._mezo_sparse_thr[id(p)] = thr.item()  # scalar
        sparse_thresholds = model._mezo_sparse_thr

    with Timer(store, "mezo_step_total", sync=True):
        with Timer(store, "mezo_perturb_pos", sync=sync_breakdown):
            _perturb_inplace_variant(params, seed, eps, +1.0, variant, sparse_thresholds, lozo_rank, quant_bits)

        with Timer(store, "mezo_forward_pos", sync=sync_breakdown):
            with torch.no_grad():
                loss_pos = model(**batch).loss.detach()

        with Timer(store, "mezo_perturb_neg", sync=sync_breakdown):
            _perturb_inplace_variant(params, seed, eps, -2.0, variant, sparse_thresholds, lozo_rank, quant_bits)

        with Timer(store, "mezo_forward_neg", sync=sync_breakdown):
            with torch.no_grad():
                loss_neg = model(**batch).loss.detach()

        with Timer(store, "mezo_restore", sync=sync_breakdown):
            _perturb_inplace_variant(params, seed, eps, +1.0, variant, sparse_thresholds, lozo_rank, quant_bits)

        with Timer(store, "mezo_loss_diff", sync=sync_breakdown):
            proj_grad = (loss_pos - loss_neg) / (2.0 * eps)

        with Timer(store, "mezo_allreduce_scalar", sync=sync_breakdown):
            if dist.is_initialized():
                dist.all_reduce(proj_grad, op=dist.ReduceOp.SUM)
                proj_grad = proj_grad / dist.get_world_size()

        with Timer(store, "mezo_update", sync=sync_breakdown):
            _perturb_inplace_variant(params, seed, eps,
                                      alpha=-lr * float(proj_grad) / eps,
                                      variant=variant, sparse_thresholds=sparse_thresholds,
                                      lozo_rank=lozo_rank, quant_bits=quant_bits)


def mezo_forward_only(model, batch, store):
    """Ablation C: just 2 forwards, no perturb/restore/update."""
    with Timer(store, "mezo_step_total", sync=True):
        with Timer(store, "mezo_forward_pos"):
            with torch.no_grad():
                _ = model(**batch).loss
        with Timer(store, "mezo_forward_neg"):
            with torch.no_grad():
                _ = model(**batch).loss


def mezo_perturb_only(model, batch, eps, lr, seed, store):
    """Ablation D: 3 perturbations + scalar update, no forward."""
    params = _iter_trainable_params(model)
    with Timer(store, "mezo_step_total", sync=True):
        with Timer(store, "mezo_perturb_pos"):
            _perturb_inplace(params, seed, eps, alpha=+1.0)
        with Timer(store, "mezo_perturb_neg"):
            _perturb_inplace(params, seed, eps, alpha=-2.0)
        with Timer(store, "mezo_restore"):
            _perturb_inplace(params, seed, eps, alpha=+1.0)
        with Timer(store, "mezo_update"):
            torch.manual_seed(seed)
            for p in params:
                z = torch.normal(mean=0.0, std=1.0, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                p.data.add_(z, alpha=-lr * 0.5)


def mezo_noop_traversal(model, store):
    """Ablation E: traverse params 4 times but do nothing — measures Python/CUDA launch overhead."""
    params = _iter_trainable_params(model)
    with Timer(store, "mezo_step_total", sync=True):
        for _ in range(4):
            with Timer(store, "noop_param_loop_pass"):
                # touch each param tensor; minimal op
                acc = 0
                for p in params:
                    acc += p.numel()  # noop, just iterate
                # Tiny GPU work to mark the boundary
                _ = torch.zeros(1, device=params[0].device).fill_(float(acc % 7))


# ----------------------------------------------------------------------------
# GRZO step (via flipout.py)
# ----------------------------------------------------------------------------

def _build_grzo_inputs(model, batch, bs):
    """Find Linear modules to expose to flipout step; same convention as trainer.py grzo path."""
    lora_modules = [
        m for m in model.modules()
        if isinstance(m, (nn.Linear, nn.LayerNorm, nn.Embedding)) and any(p.requires_grad for p in m.parameters())
    ]
    return lora_modules


def grzo_step(model, batch, eps, lr, store, sync_breakdown=True,
              variant="vanilla", lozo_rank=4, sparse_ratio=0.20, quant_bits=4, drop_s_i=False):
    """GRZO flipout step (2-sided matching production), supports 4 variants:
       - vanilla: just per-Linear U (Gaussian)
       - lozo: U = u @ v^T low-rank
       - sparse: apply per-Linear sparse mask to U
       - quzo: quantize U to quant_bits
       - drop_s_i=True (for LOZO-strict): set S=ones (no per-example column sign)
    Two-sided (matches production estimation_side=two_norm):
      - 2 forwards: enable_flip(+1) and enable_flip(-1)
      - delta = (loss_plus - loss_minus) / 2
    """
    linears = [m for m in model.modules()
               if isinstance(m, nn.Linear) and any(p.requires_grad for p in m.parameters())]
    bs = batch["input_ids"].size(0)

    with Timer(store, "grzo_step_total", sync=True):
        # 1. Sign generation (Rademacher S, R per Linear)
        with Timer(store, "grzo_sign_generation", sync=sync_breakdown):
            signs_S, signs_R = {}, {}
            for m in linears:
                w = m.weight
                signs_S[m] = ((torch.randint(0, 2, (bs, w.size(1)), device=w.device, dtype=torch.float32)) * 2 - 1).to(w.dtype)
                if drop_s_i:
                    signs_S[m] = torch.ones_like(signs_S[m])  # LOZO-strict: no column signs
                signs_R[m] = ((torch.randint(0, 2, (bs, w.size(0)), device=w.device, dtype=torch.float32)) * 2 - 1).to(w.dtype)

        # 2. U generation (variant-specific)
        with Timer(store, "grzo_U_generation", sync=sync_breakdown):
            Us = {}
            for m in linears:
                w = m.weight
                if variant == "lozo":
                    # Low-rank: U = (u @ v) / sqrt(r)
                    u = torch.randn(w.size(0), lozo_rank, device=w.device, dtype=w.dtype)
                    v = torch.randn(lozo_rank, w.size(1), device=w.device, dtype=w.dtype)
                    Us[m] = (u @ v) / (lozo_rank ** 0.5)
                else:
                    Us[m] = torch.randn(w.size(), device=w.device, dtype=w.dtype)

        # 2b. Variant-specific noise modification
        if variant == "sparse":
            with Timer(store, "grzo_sparse_mask", sync=sync_breakdown):
                # cache thresholds (small-W rule, paper-faithful)
                if not hasattr(model, '_grzo_sparse_thr'):
                    model._grzo_sparse_thr = {}
                    for m in linears:
                        n = m.weight.numel()
                        k = max(1, int(sparse_ratio * n))
                        thr = torch.kthvalue(m.weight.detach().abs().view(-1), k)[0]
                        model._grzo_sparse_thr[m] = thr
                for m in linears:
                    mask = (m.weight.detach().abs() <= model._grzo_sparse_thr[m]).to(Us[m].dtype)
                    Us[m] = Us[m] * mask

        if variant == "quzo":
            with Timer(store, "grzo_quant", sync=sync_breakdown):
                n_levels = (1 << (quant_bits - 1)) - 1
                for m in linears:
                    x = Us[m]
                    abs_max = x.detach().abs().max().clamp(min=1e-12)
                    scale = n_levels / abs_max
                    Us[m] = ((x * scale).round().clamp(-n_levels, n_levels) / scale).to(x.dtype)

        # 3. Hook install (sign-aware: direction +1 first pass, -1 second pass)
        flip_dir = [+1.0]
        hooks = []
        with Timer(store, "grzo_hook_install", sync=sync_breakdown):
            def make_hook(U, S, R):
                def hook(module, inp, out):
                    x = inp[0]
                    if x.dim() == 3:
                        x_mod = x * S.unsqueeze(1)
                        pert = torch.nn.functional.linear(x_mod, U)
                        pert = pert * R.unsqueeze(1)
                    else:
                        x_mod = x * S
                        pert = torch.nn.functional.linear(x_mod, U)
                        pert = pert * R
                    return out + (eps * flip_dir[0]) * pert
                return hook
            for m in linears:
                hooks.append(m.register_forward_hook(make_hook(Us[m], signs_S[m], signs_R[m])))

        # 4a. Forward with +eps perturbation
        with Timer(store, "grzo_forward_plus", sync=sync_breakdown):
            flip_dir[0] = +1.0
            with torch.no_grad():
                outputs_p = model(**batch)
                losses_plus = outputs_p.logits.float().mean(dim=tuple(range(1, outputs_p.logits.dim())))

        # 4b. Forward with -eps perturbation (2nd forward, matches production two_norm)
        with Timer(store, "grzo_forward_minus", sync=sync_breakdown):
            flip_dir[0] = -1.0
            with torch.no_grad():
                outputs_m = model(**batch)
                losses_minus = outputs_m.logits.float().mean(dim=tuple(range(1, outputs_m.logits.dim())))

        # 5. delta + group_relative_norm
        with Timer(store, "grzo_group_relative_norm", sync=sync_breakdown):
            delta = (losses_plus - losses_minus) / 2.0
            mu = delta.mean()
            sigma_l = delta.std().clamp(min=1e-8)
            adv = (delta - mu) / sigma_l  # (bs,)

        # provide a uniform "forward_with_flipout" stat = plus + minus
        fp = store["grzo_forward_plus"][-1] if "grzo_forward_plus" in store else 0
        fm = store["grzo_forward_minus"][-1] if "grzo_forward_minus" in store else 0
        store.setdefault("grzo_forward_total", []).append(fp + fm)

        # 6. Hook remove
        with Timer(store, "grzo_hook_remove", sync=sync_breakdown):
            for h in hooks:
                h.remove()

        # 7. All-reduce advantage (DDP)
        with Timer(store, "grzo_allreduce_adv", sync=sync_breakdown):
            if dist.is_initialized():
                dist.all_reduce(adv, op=dist.ReduceOp.SUM)
                adv = adv / dist.get_world_size()

        # 8. Update: weight -= lr * sum_i adv_i * (U * outer(R_i, S_i)) → ONE U/Linear, no in-place per-param
        with Timer(store, "grzo_update", sync=sync_breakdown):
            for m in linears:
                # Aggregate sign outer products weighted by adv: shape (dout, din)
                # adv (bs,), R (bs, dout), S (bs, din)
                weighted_R = signs_R[m] * adv.unsqueeze(1).to(signs_R[m].dtype)  # (bs, dout)
                outer = weighted_R.t() @ signs_S[m]  # (dout, din)
                m.weight.data.add_(Us[m] * outer, alpha=-lr / bs)


def grzo_forward_no_flipout(model, batch, store):
    """Ablation F: 1 forward with NO perturbation hooks — pure base forward time."""
    with Timer(store, "grzo_step_total", sync=True):
        with Timer(store, "grzo_forward_base"):
            with torch.no_grad():
                _ = model(**batch).loss


def grzo_sign_only(model, batch, store):
    """Ablation G: simulate sign/U generation cost without forward."""
    lora_modules = _build_grzo_inputs(model, batch, batch["input_ids"].size(0))
    bs = batch["input_ids"].size(0)
    with Timer(store, "grzo_step_total", sync=True):
        with Timer(store, "grzo_sign_generation"):
            for m in lora_modules:
                if isinstance(m, nn.Linear):
                    g = torch.Generator(device=m.weight.device)
                    g.manual_seed(42)
                    _ = torch.randint(0, 2, (bs, m.weight.size(0)),
                                       generator=g, device=m.weight.device, dtype=torch.float32)
                    _ = torch.randint(0, 2, (bs, m.weight.size(1)),
                                       generator=g, device=m.weight.device, dtype=torch.float32)
        with Timer(store, "grzo_U_generation"):
            for m in lora_modules:
                if isinstance(m, nn.Linear):
                    g = torch.Generator(device=m.weight.device)
                    g.manual_seed(123)
                    _ = torch.randn(m.weight.size(), generator=g, device=m.weight.device, dtype=m.weight.dtype)


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate_method(method_name, all_rank_data):
    """For one method, aggregate per-rank data into summary stats."""
    # all_rank_data: {rank: {range_name: [t_ms, ...]}}
    summary = []
    range_names = set()
    for r in all_rank_data.values():
        range_names.update(r.keys())
    for rname in sorted(range_names):
        # collect all timings across all steps and ranks
        all_ts = []
        per_rank_means = []
        for rank, store in all_rank_data.items():
            ts = store.get(rname, [])
            if ts:
                per_rank_means.append(sum(ts) / len(ts))
                all_ts.extend(ts)
        if not all_ts:
            continue
        sorted_ts = sorted(all_ts)
        n = len(sorted_ts)
        p50 = sorted_ts[n // 2]
        p90 = sorted_ts[max(0, int(n * 0.9) - 1)]
        mean = statistics.mean(all_ts)
        std = statistics.stdev(all_ts) if n > 1 else 0.0
        max_v = max(all_ts)
        rank_max_mean = max(per_rank_means) if per_rank_means else 0.0
        summary.append({
            "method": method_name,
            "range": rname,
            "mean_ms": round(mean, 3),
            "std_ms": round(std, 3),
            "p50_ms": round(p50, 3),
            "p90_ms": round(p90, 3),
            "max_ms": round(max_v, 3),
            "rank_max_mean_ms": round(rank_max_mean, 3),
            "n_samples": n,
            "n_ranks": len(all_rank_data),
        })
    return summary


def write_report(summaries, output_dir, env_info):
    """Write a markdown report with cross-method comparison."""
    lines = ["# MeZO vs GRZO wall-clock profiling\n"]
    lines.append("## Environment\n")
    for k, v in env_info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Total step time table
    lines.append("\n## Total step time per method\n")
    lines.append("| Method | mean (ms) | std | p50 | p90 | max |")
    lines.append("|---|---|---|---|---|---|")
    for method, rows in summaries.items():
        tot = next((r for r in rows if r["range"].endswith("step_total")), None)
        if tot:
            lines.append(f"| **{method}** | {tot['mean_ms']:.1f} | {tot['std_ms']:.1f} | {tot['p50_ms']:.1f} | {tot['p90_ms']:.1f} | {tot['max_ms']:.1f} |")

    # Per-method breakdown
    for method, rows in summaries.items():
        lines.append(f"\n## {method} breakdown\n")
        lines.append("| Range | mean (ms) | p50 | p90 | % of total |")
        lines.append("|---|---|---|---|---|")
        tot = next((r for r in rows if r["range"].endswith("step_total")), None)
        tot_ms = tot["mean_ms"] if tot else 1
        for r in rows:
            pct = 100.0 * r["mean_ms"] / tot_ms if tot_ms > 0 else 0.0
            lines.append(f"| {r['range']} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | {r['p90_ms']:.2f} | {pct:.1f}% |")

    # Findings
    lines.append("\n## Quick findings\n")
    if "mezo" in summaries and "grzo" in summaries:
        m_tot = next((r["mean_ms"] for r in summaries["mezo"] if r["range"] == "mezo_step_total"), 0)
        g_tot = next((r["mean_ms"] for r in summaries["grzo"] if r["range"] == "grzo_step_total"), 0)
        if g_tot > 0:
            lines.append(f"- MeZO/GRZO ratio: **{m_tot/g_tot:.2f}×**")
        # MeZO perturb fraction
        m_pert = sum(r["mean_ms"] for r in summaries["mezo"] if r["range"] in ("mezo_perturb_pos","mezo_perturb_neg","mezo_restore","mezo_update"))
        m_fwd = sum(r["mean_ms"] for r in summaries["mezo"] if r["range"] in ("mezo_forward_pos","mezo_forward_neg"))
        if m_tot > 0:
            lines.append(f"- MeZO perturb+restore+update: **{m_pert:.1f} ms** ({100*m_pert/m_tot:.0f}% of step)")
            lines.append(f"- MeZO 2× forward: **{m_fwd:.1f} ms** ({100*m_fwd/m_tot:.0f}% of step)")

    out = output_dir / "report.md"
    out.write_text("\n".join(lines))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def make_fake_batch(tokenizer, bs, seq_len, device):
    """Build a synthetic input batch."""
    text = "Question: Is this sentence positive? Answer:"
    enc = tokenizer([text] * bs, return_tensors="pt", padding="max_length",
                    max_length=seq_len, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    # Use input_ids as labels for loss computation
    enc["labels"] = enc["input_ids"].clone()
    return enc


def run_method(method_name, model, batch, eps, lr, warmup, profile, rank, world_size):
    """Run warmup + profile steps for one method, return per-step timings."""
    store = {}
    for step in range(warmup + profile):
        # measurement starts after warmup
        record = step >= warmup
        local_store = store if record else {}

        if method_name == "mezo":
            mezo_step(model, batch, eps, lr, seed=42 + step, store=local_store)
        elif method_name == "mezo_sparse":
            mezo_variant_step(model, batch, eps, lr, 42 + step, local_store, variant="sparse")
        elif method_name == "mezo_lozo":
            mezo_variant_step(model, batch, eps, lr, 42 + step, local_store, variant="lozo")
        elif method_name == "mezo_quzo":
            mezo_variant_step(model, batch, eps, lr, 42 + step, local_store, variant="quzo")
        elif method_name == "grzo":
            grzo_step(model, batch, eps, lr, store=local_store, variant="vanilla")
        elif method_name == "grzo_sparse":
            grzo_step(model, batch, eps, lr, store=local_store, variant="sparse")
        elif method_name == "grzo_lozo":
            grzo_step(model, batch, eps, lr, store=local_store, variant="lozo")
        elif method_name == "grzo_lozo_strict":
            grzo_step(model, batch, eps, lr, store=local_store, variant="lozo", drop_s_i=True)
        elif method_name == "grzo_quzo":
            grzo_step(model, batch, eps, lr, store=local_store, variant="quzo")
        elif method_name == "mezo_forward_only":
            mezo_forward_only(model, batch, store=local_store)
        elif method_name == "mezo_perturb_only":
            mezo_perturb_only(model, batch, eps, lr, seed=42 + step, store=local_store)
        elif method_name == "mezo_noop_traversal":
            mezo_noop_traversal(model, store=local_store)
        elif method_name == "grzo_forward_no_flipout":
            grzo_forward_no_flipout(model, batch, store=local_store)
        elif method_name == "grzo_sign_only":
            grzo_sign_only(model, batch, store=local_store)
        else:
            raise ValueError(f"unknown method: {method_name}")

    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="mezo,grzo,mezo_forward_only,mezo_perturb_only,mezo_noop_traversal,grzo_forward_no_flipout,grzo_sign_only",
                    help="comma-separated method names")
    ap.add_argument("--model", required=True, help="local path or HF id of the model")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--profile-steps", type=int, default=20)
    ap.add_argument("--output", default="profiling/breakdown")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-7)
    ap.add_argument("--torch-profiler", action="store_true")
    args = ap.parse_args()

    # DDP init
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        dist.init_process_group("nccl")

    if rank == 0:
        print(f"[rank0] world_size={world_size} model={args.model} bs={args.batch_size} seq={args.seq_len}")
        print(f"[rank0] warmup={args.warmup_steps} profile={args.profile_steps} methods={args.methods}")

    # Load model + tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)  # ZO still needs requires_grad to identify perturbable params

    # Note: GRZO step here uses a MINIMAL replicated version (sign+U gen, hook forward,
    # update via U @ outer(R, S)) rather than grzo_flipout_step_ddp, to avoid needing the
    # per-example-loss model wrapper. Timing of each component is what matters for profiling.

    batch = make_fake_batch(tok, args.batch_size, args.seq_len, device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_summaries = {}

    for method in methods:
        if dist.is_initialized():
            dist.barrier()
        # Clear per-method state + free GPU memory between methods
        for attr in ("_mezo_sparse_thr", "_mezo_sparse_masks", "_grzo_sparse_thr"):
            if hasattr(model, attr):
                delattr(model, attr)
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"\n[rank0] === profiling {method} === free_GB={torch.cuda.mem_get_info()[0]/1e9:.1f}")
        t0 = time.time()
        store = run_method(method, model, batch, args.eps, args.lr,
                           args.warmup_steps, args.profile_steps, rank, world_size)
        if dist.is_initialized():
            dist.barrier()
        wall = time.time() - t0
        if rank == 0:
            print(f"[rank0] {method} wall: {wall:.1f}s")

        # Write per-rank JSONL
        out_path = output_dir / f"{method}_{ts}_rank{rank}.jsonl"
        with open(out_path, "w") as f:
            for rname, ts_list in store.items():
                for i, t in enumerate(ts_list):
                    f.write(json.dumps({"step": i, "range": rname, "t_ms": t, "rank": rank}) + "\n")

        # Gather to rank 0 for aggregation
        all_rank_data = {rank: store}
        if dist.is_initialized():
            gathered = [None] * world_size
            dist.all_gather_object(gathered, store)
            if rank == 0:
                all_rank_data = {i: gathered[i] for i in range(world_size)}

        if rank == 0:
            all_summaries[method] = aggregate_method(method, all_rank_data)

    # Final outputs (rank 0 only)
    if rank == 0:
        # CSV
        csv_path = output_dir / f"summary_{ts}.csv"
        with open(csv_path, "w") as f:
            f.write("method,range,mean_ms,std_ms,p50_ms,p90_ms,max_ms,rank_max_mean_ms,n_samples,n_ranks\n")
            for method, rows in all_summaries.items():
                for r in rows:
                    f.write(f"{method},{r['range']},{r['mean_ms']},{r['std_ms']},{r['p50_ms']},{r['p90_ms']},{r['max_ms']},{r['rank_max_mean_ms']},{r['n_samples']},{r['n_ranks']}\n")

        # Report
        env_info = {
            "GPU type": torch.cuda.get_device_name(0),
            "world_size": world_size,
            "model": args.model.split("/")[-1],
            "dtype": "fp16",
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "seq_len": args.seq_len,
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "timestamp": ts,
        }
        report_path = write_report(all_summaries, output_dir, env_info)
        print(f"\n[rank0] outputs:")
        print(f"  {csv_path}")
        print(f"  {report_path}")
        print(f"  per-rank JSONL: {output_dir}/*_{ts}_rank*.jsonl")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
