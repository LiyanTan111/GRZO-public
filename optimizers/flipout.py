import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
except ImportError:
    dist = None


class NormNoiseState:
    """Simplified state for LN/RMSNorm: per-example independent +/-1 vectors."""
    def __init__(self, bs, dim, seed=None):
        self.bs = bs
        self.dim = dim
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        self.seed = int(seed)
        self.sigma_sign = +1

class EmbNoiseState:
    """Simplified state for Embedding: row-wise sparse perturbation with hash-based +/-1."""
    def __init__(self, bs, seed_u=None, seed_s=None):
        self.bs = bs
        if seed_u is None:
            seed_u = torch.randint(0, 2**32, (1,)).item()
        if seed_s is None:
            seed_s = torch.randint(0, 2**32, (1,)).item()
        self.seed_u = int(seed_u)   # controls u_row vectors
        self.seed_s = int(seed_s)   # controls scalar signs per occurrence
        self.sigma_sign = +1
        # cache mapping from forward for correct update (multiple embeddings)
        self.last_ids: torch.Tensor | None = None
        self.last_uniq: torch.Tensor | None = None
        self.last_inv: torch.Tensor | None = None
        self.last_ex_ids: torch.Tensor | None = None

class PrefixParamState:
    """State for prefix_keys/prefix_values: shared-noise direct parameter perturbation.
    Unlike Linear/LoRA modules, prefix params are not processed by a module forward pass,
    so we perturb them directly (MeZO-style +/-) and use the GRZO two_norm estimator on
    per-example losses to recover the gradient signal.
    """
    def __init__(self, z_keys=None, z_values=None):
        self.z_keys = z_keys      # noise tensor, same shape as prefix_keys
        self.z_values = z_values  # noise tensor, same shape as prefix_values
        self.sigma_sign = +1

def _rademacher_2d(rows, cols, seed, device):
    """Deterministic +/-1 matrix using simple hash.
    Returns float32 tensor in {+1, -1} of shape (rows, cols).
    """
    r = torch.arange(rows, device=device, dtype=torch.int64).view(rows, 1)
    c = torch.arange(cols, device=device, dtype=torch.int64).view(1, cols)
    x = r * 1103515245 + c * 12345 + seed
    return ((x & 1) * 2 - 1).to(torch.float32)

def _rademacher_scalar(ex_ids, token_ids, seed, device):
    """Generate +/-1 signs via hash for per-occurrence scalar.
    ex_ids, token_ids: int64 same shape [N]
    """
    x = token_ids * 1103515245 + ex_ids * 12345 + seed
    return ((x & 1) * 2 - 1).to(torch.float32).to(device)

def _rademacher_rows_dims(row_ids, D, seed, device):
    """Generate +/-1 vectors for specific rows.
    row_ids: (K,) int64 -> returns (K,D) +/-1 float32
    """
    rid = row_ids.to(torch.int64).view(-1, 1)
    dim = torch.arange(D, device=device, dtype=torch.int64).view(1, D)
    x = rid * 1103515245 + dim * 12345 + seed
    return ((x & 1) * 2 - 1).to(torch.float32)

def _normed_x(module, x):
    """Compute normalized x for LayerNorm/RMSNorm without weight/bias."""
    if isinstance(module, nn.LayerNorm):
        return F.layer_norm(x, module.normalized_shape, weight=None, bias=None, eps=module.eps)
    # RMSNorm / LlamaRMSNorm
    eps = getattr(module, "eps", None)
    if eps is None:
        eps = getattr(module, "variance_epsilon", 1e-6)
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms

def _zo_quant_dequant_sym(x, nbits, generator=None):
    """Per-tensor symmetric STOCHASTIC quantize+dequantize at nbits precision.
    Implements QuZO (Zhou et al. 2025) Eq. 5: floor + Bernoulli up-rounding.
    Unbiased: E[Q(x)] = x.

    `generator`: torch.Generator for Bernoulli randomness. If None, uses
    PyTorch's default RNG state. Caller passes distinct generators for the
    forward-perturbation copy (u_{i,1}) and the gradient copy (u_{i,2}) so
    they are conditionally independent — see QuZO Eq. 7-10.
    """
    if nbits is None or nbits >= 32:
        return x
    if nbits == 1:
        return torch.sign(x).to(x.dtype)
    n_levels = (1 << (nbits - 1)) - 1  # 2-bit -> 1, 3-bit -> 3, 4-bit -> 7, 8-bit -> 127
    abs_max = x.detach().abs().max().clamp(min=1e-12)
    scale = n_levels / abs_max
    scaled = x.float() * scale
    if generator is None:
        # No generator → fall back to deterministic rounding (old behavior).
        # Used by paths (e.g. mezo_quzo) that regenerate z from a seed multiple
        # times per step and need the same quantization each time.
        q = scaled.round().clamp(-n_levels, n_levels)
    else:
        # Stochastic rounding per QuZO Eq. 5.
        floor_val = scaled.floor()
        frac = (scaled - floor_val).clamp(0.0, 1.0)
        rand = torch.empty_like(frac).uniform_(0.0, 1.0, generator=generator)
        rounded = floor_val + (rand < frac).to(scaled.dtype)
        q = rounded.clamp(-n_levels, n_levels)
    return (q / scale).to(x.dtype)


def generate_noise(shape, generator, dtype, device, distribution="gaussian"):
    """Generate noise with specified distribution.
    
    Args:
        shape: Shape of noise tensor
        generator: PyTorch random generator
        dtype: Data type
        device: Device
        distribution: "gaussian" for N(0,1) or "rademacher" for {-1, +1}
    """
    if distribution == "rademacher":
        # Rademacher: uniform {-1, +1}
        return (torch.randint(0, 2, shape, generator=generator, device=device, dtype=torch.int8) * 2 - 1).to(dtype)
    else:
        # Gaussian: N(0, 1)
        return torch.randn(shape, generator=generator, dtype=dtype, device=device)


def _expand_grzo_advantages(adv, bs_flat, bs_eff, inputs):
    """Expand per-example advantages to flattened option batches when needed."""
    if (
        adv.shape[0] == bs_eff
        and bs_flat != bs_eff
        and "num_options" in inputs
        and inputs["num_options"] is not None
    ):
        adv_list = []
        for i, n_opt in enumerate(inputs["num_options"]):
            adv_list.extend([adv[i].item()] * int(n_opt))
        return torch.tensor(adv_list, dtype=adv.dtype, device=adv.device)
    return adv


def _compute_grzo_response_weights(
    loss_plus,
    loss_minus,
    estimation_side,
    eps,
    sigma,
    bs_flat,
    bs_eff,
    inputs,
    adv_std_floor: float = 0.0,
    adv_clip: float = 0.0,
):
    """Compute GRZO-style response weights and return-loss bookkeeping.

    adv_std_floor: if > 0, clamp the denominator std to at least this value
        before normalizing — prevents weak-signal batches from amplifying noise.
    adv_clip: if > 0, clip the resulting per-example advantages to [-c, +c].
    """
    assert loss_plus.shape[0] in (bs_flat, bs_eff), (
        f"loss_plus shape {loss_plus.shape[0]} not in "
        f"(bs_flat={bs_flat}, bs_eff={bs_eff})"
    )

    if estimation_side == "one":
        losses = loss_plus.float()
        mu_L = losses.mean()
        sigma_L = losses.std()
        if adv_std_floor > 0:
            sigma_L = torch.clamp(sigma_L, min=adv_std_floor)
        adv = (losses - mu_L) / (sigma_L + eps)
        divisor = sigma
        return_loss = loss_plus.mean().item()
    elif estimation_side == "uniform":
        adv = torch.ones_like(loss_plus.float())
        divisor = sigma
        return_loss = loss_plus.mean().item()
    elif estimation_side in ("two", "two_norm"):
        assert loss_minus is not None, "loss_minus is required for two-sided estimation"
        assert loss_minus.shape[0] in (bs_flat, bs_eff), (
            f"loss_minus shape {loss_minus.shape[0]} not in "
            f"(bs_flat={bs_flat}, bs_eff={bs_eff})"
        )
        delta = (loss_plus.float() - loss_minus.float()) / 2.0
        if estimation_side == "two_norm":
            sigma_delta = delta.std()
            if adv_std_floor > 0:
                sigma_delta = torch.clamp(sigma_delta, min=adv_std_floor)
            adv = delta / (sigma_delta + eps)
        else:
            adv = delta
        divisor = 2.0 * sigma
        return_loss = (loss_plus.mean().item() + loss_minus.mean().item()) / 2.0
    else:
        raise ValueError(f"Unsupported estimation_side: {estimation_side}")

    if adv_clip > 0:
        adv = adv.clamp(min=-adv_clip, max=adv_clip)

    adv_expanded = _expand_grzo_advantages(adv, bs_flat, bs_eff, inputs)
    return adv_expanded, divisor, return_loss


class FlipoutState:
    # one per module, per training step
    def __init__(self, bs, shape, device, dtype, is_lora=True, u_distribution="gaussian"):
        self.is_lora = is_lora
        self.u_distribution = u_distribution

        if is_lora:
            # LoRA case: shape is (A_shape, B_shape)
            A_shape, B_shape = shape
            r, din = A_shape
            dout, r2 = B_shape
            assert r2 == r

            # base noise (shared across batch) - allocate directly
            # Use specified distribution (gaussian or rademacher)
            gen_UA = torch.Generator(device=device)
            gen_UA.manual_seed(int(torch.randint(0, 2**32, (1,)).item()))
            gen_UB = torch.Generator(device=device)
            gen_UB.manual_seed(int(torch.randint(0, 2**32, (1,)).item()))
            self.UA = generate_noise((r, din), gen_UA, dtype, device, u_distribution)
            self.UB = generate_noise((dout, r), gen_UB, dtype, device, u_distribution)

            # Rademacher signs per example
            self.RA = torch.randint(0, 2, (bs, r), device=device, dtype=torch.int8) * 2 - 1
            self.SA = torch.randint(0, 2, (bs, din), device=device, dtype=torch.int8) * 2 - 1
            self.RB = torch.randint(0, 2, (bs, dout), device=device, dtype=torch.int8) * 2 - 1
            self.SB = torch.randint(0, 2, (bs, r), device=device, dtype=torch.int8) * 2 - 1
        else:
            # Full FT case: shape can be 2D (Linear), 1D (LayerNorm), or 2D (Embedding)
            if len(shape) == 2:
                # Linear or Embedding: (dout, din) or (num_embeddings, embedding_dim)
                dout, din = shape

                # Keep only a seed for U (regenerated on demand to save memory)
                self.seed = torch.randint(0, 2**32, (1,)).item()
                # Rademacher signs for 2D weight matrices
                self.RA = torch.randint(0, 2, (bs, dout), device=device, dtype=torch.int8) * 2 - 1
                self.SA = torch.randint(0, 2, (bs, din), device=device, dtype=torch.int8) * 2 - 1
            elif len(shape) == 1:
                # LayerNorm/RMSNorm: (D,)
                D = shape[0]
                
                # Use seed to save memory for U
                self.seed = torch.randint(0, 2**32, (1,)).item()
                
                # For 1D parameters, use element-wise noise
                # R and S both have shape (bs, D) for element-wise perturbation
                self.RA = torch.randint(0, 2, (bs, D), device=device, dtype=torch.int8) * 2 - 1
                self.SA = torch.randint(0, 2, (bs, D), device=device, dtype=torch.int8) * 2 - 1
            else:
                raise ValueError(f"Unsupported parameter shape: {shape}")

        # will be set to +1 or -1 for the two passes
        self.sigma_sign = +1


def compute_per_example_loss(model, batch):
    """
    Computes loss per example in the batch.
    """
    # Get device from model parameters
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    
    # Pass special argument to hint modified forward 
    # (if model is wrapped with forward_wrap_with_option_len)
    outputs = model(**inputs, return_per_example_loss=True)
    
    # If the model returned a tensor directly (custom forward might return loss tensor), use it.
    if isinstance(outputs, torch.Tensor):
        loss = outputs
    elif hasattr(outputs, "loss"):
        loss = outputs.loss
    else:
        # Fallback if outputs is tuple or unknown (assume scalar/tensor at 0)
        loss = outputs[0]

    # If loss appears scalar (mean reduced) but we wanted per-example:
    # This shouldn't happen if our modification to utils.py works and model uses that wrapper.
    # If the model is standard DataParallel or similar, unwrapping might be implicitly needed but usually "**inputs" works.
    
    # Ensure (bs,)
    if loss.dim() == 0:
        # If scalar, we failed to get per-example loss. 
        # But this code is "pseudo-code" made real. 
        # If we failed, we probably can't recover.
        # But maybe we return scalar to not crash, though Flipout will be wrong (zero grad).
        pass
        
    return loss

@torch.no_grad()
def grzo_flipout_step(model, inputs, lora_modules, sigma, lr, eps=1e-8, u_distribution="gaussian", estimation_side="two",
                       accumulate_grad=False, grad_accumulation_storage=None, accum_step=1, total_accum_steps=1,
                       sparse_masks=None, lozo_uv=None, quant_bits=None,
                       adv_std_floor: float = 0.0, adv_clip: float = 0.0, lozo_strict: bool = False):
    """
    lora_modules: list of LoRALinear modules OR nn.Linear modules (for full FT)
    inputs: dict of tensors, batch inputs
    u_distribution: "gaussian" (default) or "rademacher" for base noise U
    estimation_side: "one" for single-sided estimation or "two" for two-sided estimation
    accumulate_grad: if True, accumulate gradients without applying update
    grad_accumulation_storage: dict to store accumulated gradients
    accum_step: current accumulation step (1-indexed)
    total_accum_steps: total number of accumulation steps
    sparse_masks: optional dict {module: bool_tensor of shape m.weight.shape}.
        When provided, the U noise (and thus the perturbation DTheta and gradient)
        is masked elementwise so only "mask=True" weights are perturbed/updated.
        This implements GRZO + SparseMeZO. Only used for Linear modules in full
        FT mode; LoRA / Embedding / LayerNorm modules are unaffected.
    lozo_uv: optional dict {module: (u, v)} where u is (dout, r), v is (r, din)
        Gaussian. When present, the base noise U for that Linear module is
        replaced by "u @ v" (a rank-r tensor in the weight shape) instead of a
        full-rank Gaussian. Implements GRZO + LOZO. The caller is responsible
        for refreshing (u, v) according to the LOZO schedule (v every K steps,
        u every step).
    quant_bits: int or None. When set (e.g. 2, 4, 8) the base noise U used in
        BOTH the forward hook and the gradient update is quantized+dequantized
        to nbits precision via _zo_quant_dequant_sym. Implements GRZO + QuZO
        (perturbation-quantization branch). Weights themselves stay fp16/fp32.
    """

    bs_flat = inputs["input_ids"].shape[0] # flattened batch size (e.g., 44 for multi-choice)
    # Derive device from model parameters (inputs may still be on CPU at this point)
    device = next(model.parameters()).device
    # Get dtype from first param
    dtype = next(model.parameters()).dtype
    
    # Calculate effective batch size for multi-choice tasks
    if "num_options" in inputs and inputs["num_options"] is not None:
        bs_eff = len(inputs["num_options"])  # actual number of examples (e.g., 7)
    else:
        bs_eff = bs_flat  # standard case, no flattening

    # Determine mode: LoRA or Full FT based on the first module type
    # (Assuming homogeneous list)
    is_lora = False
    if len(lora_modules) > 0:
        first_module = lora_modules[0]
        # Check if any module is a LoRA module (has lora_A attribute)
        if hasattr(first_module, "lora_A"):
            is_lora = True

    # 1) build flipout states using flattened batch size for noise injection
    # Note: R/S are initialized with bs_flat (total options for multi-choice, equals bs_eff for standard tasks)
    # This ensures each forward pass example/option gets independent perturbations
    # For multi-choice: each option gets its own R/S noise, advantages are per-question (see line 271-283)
    states = {}
    for m in lora_modules:
        if hasattr(m, 'prefix_keys') or hasattr(m, 'prefix_values'):
            # Prefix parameter module: generate independent shared noise for keys and values
            rng = torch.Generator(device=device)
            z_keys = z_values = None
            if hasattr(m, 'prefix_keys') and m.prefix_keys.requires_grad:
                rng.manual_seed(torch.randint(0, 2**32, (1,)).item())
                z_keys = generate_noise(m.prefix_keys.shape, rng, dtype, device, u_distribution)
            if hasattr(m, 'prefix_values') and m.prefix_values.requires_grad:
                rng.manual_seed(torch.randint(0, 2**32, (1,)).item())
                z_values = generate_noise(m.prefix_values.shape, rng, dtype, device, u_distribution)
            states[m] = PrefixParamState(z_keys=z_keys, z_values=z_values)
        elif is_lora:
            A_shape = tuple(m.lora_A.shape)  # (r, din)
            B_shape = tuple(m.lora_B.shape)  # (dout, r)
            states[m] = FlipoutState(bs_flat, (A_shape, B_shape), device=device, dtype=dtype, is_lora=True, u_distribution=u_distribution)
        else:
            # Full FT: determine parameter shape based on module type
            if isinstance(m, nn.Embedding):
                # Embedding: use simplified sparse state
                states[m] = EmbNoiseState(bs=bs_flat)
            elif isinstance(m, (nn.LayerNorm,)) or type(m).__name__ in ['LlamaRMSNorm', 'RMSNorm']:
                # LayerNorm/RMSNorm: use simplified per-example noise state
                dim = m.weight.shape[0]
                states[m] = NormNoiseState(bs=bs_flat, dim=dim)
            else:
                # Linear and others: weight shape is (dout, din)
                shape = tuple(m.weight.shape)
                states[m] = FlipoutState(bs_flat, shape, device=device, dtype=dtype, is_lora=False, u_distribution=u_distribution)
                # Strict LOZO-GRZO: drop per-example column sign s_i so update
                # stays in span(V) — see paper "LOZO-GRZO" combination.
                if lozo_strict and lozo_uv is not None and m in lozo_uv:
                    states[m].SA = torch.ones_like(states[m].SA)

    # Hooks for Full FT
    # We define a forward hook to inject noise
    hooks = []
    prefix_perturbed = []  # (module, 'keys'/'values', original_data) for direct-param restore
    
    def make_embedding_hook_simple(st: EmbNoiseState, sigma: float):
        """Forward hook for nn.Embedding: out += sign*sigma * u_row[token] * s(ex, token)
        only for rows that appear in batch."""
        def hook(module, inputs, output):
            # Skip if weights are on meta device (during initialization)
            if module.weight.device.type == 'meta':
                return output
            
            ids = inputs[0]  # (B,S) or (B,)
            out = output

            if ids.dim() == 1:
                ids = ids.unsqueeze(1)        # (B,) -> (B, 1)
                out = out.unsqueeze(1)        # (B, D) -> (B, 1, D)
                squeeze_back = True
            else:
                squeeze_back = False

            B, S, D = out.shape
            device = ids.device
            dtype = out.dtype
            assert B == st.bs, f"Embedding batch mismatch: got B={B}, expected {st.bs}"

            ids_flat = ids.reshape(-1).to(torch.int64)  # (Nocc,)
            ex_ids = torch.arange(B, device=device, dtype=torch.int64).view(B,1).expand(B,S).reshape(-1)

            uniq, inv = torch.unique(ids_flat, sorted=True, return_inverse=True)  # uniq: (K,), inv:(Nocc,)

            u_rows = _rademacher_rows_dims(uniq, D, st.seed_u, device).to(dtype)  # (K,D)
            s_occ = _rademacher_scalar(ex_ids, ids_flat, st.seed_s, device).to(dtype)  # (Nocc,)

            noise = u_rows[inv] * s_occ[:, None]  # (Nocc,D)
            eps_eff = float(st.sigma_sign) * float(sigma)
            out = out + eps_eff * noise.view(B, S, D)

            # cache for update
            st.last_ids = ids.detach()
            st.last_uniq = uniq
            st.last_inv = inv
            st.last_ex_ids = ex_ids

            if squeeze_back:
                out = out.squeeze(1)
            return out
        return hook
    
    def make_norm_hook_simple(st: NormNoiseState, sigma: float):
        """Forward hook for LN/RMSNorm: out += sign*sigma * norm(x) * u
        where u is per-example independent +/-1 vector (B, D), generated by hash."""
        def hook(module, inputs, output):
            # Skip if weights are on meta device (during initialization)
            if module.weight.device.type == 'meta':
                return output
            
            x = inputs[0]
            D = x.shape[-1]
            bs_state = st.bs
            dtype = output.dtype
            device = x.device

            eps_eff = float(st.sigma_sign) * float(sigma)

            # Case A: x is (B, ..., D)
            if x.shape[0] == bs_state:
                B = bs_state
                xhat = _normed_x(module, x).to(dtype)
                u = _rademacher_2d(B, D, st.seed, device).to(dtype)  # (B,D)
                while u.dim() < xhat.dim():
                    u = u.unsqueeze(1)
                return output + eps_eff * (xhat * u)

            # Case B: x is (B*S, D) flattened
            if x.dim() == 2 and x.shape[0] % bs_state == 0:
                S = x.shape[0] // bs_state
                x2 = x.view(bs_state, S, D)
                out2 = output.view(bs_state, S, D)
                xhat = _normed_x(module, x2).to(dtype)
                u = _rademacher_2d(bs_state, D, st.seed, device).to(dtype).view(bs_state, 1, D)
                out2 = out2 + eps_eff * (xhat * u)
                return out2.view(x.shape[0], D)

            # Fallback: do nothing (or raise)
            return output
        return hook
    
    def get_flipout_hook(st, sigma):
        def hook(module, input, output):
            # Skip if weights are on meta device (during initialization)
            if module.weight.device.type == 'meta':
                return output

            # This hook is only for Linear layers
            # Embedding and LayerNorm/RMSNorm use specialized hooks

            x = input[0]

            # GRZO + LOZO: replace U with rank-r "u @ v" from the caller's cache.
            if lozo_uv is not None and module in lozo_uv:
                _u, _v = lozo_uv[module]
                _r_norm = float(_u.shape[1]) ** 0.5  # rank
                U = (_u.to(device=module.weight.device, dtype=module.weight.dtype)
                     @ _v.to(device=module.weight.device, dtype=module.weight.dtype)) / _r_norm
            else:
                # Generate U on the fly
                rng = torch.Generator(device=module.weight.device)
                rng.manual_seed(st.seed)
                U = generate_noise(module.weight.shape, rng, module.weight.dtype, module.weight.device, st.u_distribution)

            # GRZO + QuZO: stochastic-rounding quantization of U for the forward
            # perturbation (u_{i,1}). The gradient uses an independently rounded
            # copy (u_{i,2}) so the outer product is unbiased (QuZO Eq. 7-10).
            if quant_bits is not None:
                _q_rng = torch.Generator(device=U.device)
                _q_rng.manual_seed(int(st.seed) ^ 0xF0B1A11)
                U = _zo_quant_dequant_sym(U, quant_bits, generator=_q_rng)

            # GRZO + SparseMeZO: apply sparse mask to U so that only mask=True
            # entries of the weight matrix get perturbed in the forward pass.
            if sparse_masks is not None and module in sparse_masks:
                _mask = sparse_masks[module].to(device=U.device, dtype=U.dtype)
                U = U * _mask

            bs_state = st.RA.shape[0]
            SA_ = st.SA.to(device=x.device, dtype=x.dtype)
            RA_ = st.RA.to(device=x.device, dtype=x.dtype)

            # Handling flattened input (Batch*Seq, Dim)
            if x.dim() == 2 and x.shape[0] != bs_state and x.shape[0] % bs_state == 0:
                seq_len = x.shape[0] // bs_state
                x_reshaped = x.view(bs_state, seq_len, -1)
                SA_ = SA_.view(bs_state, 1, -1)
                RA_ = RA_.view(bs_state, 1, -1)
                term1 = x_reshaped * SA_
                res = term1 @ U.T
                noise = res * RA_
                noise = noise.view(x.shape[0], -1)
            elif x.dim() > 2:
                SA_ = SA_.view(bs_state, 1, -1)
                RA_ = RA_.view(bs_state, 1, -1)
                term1 = x * SA_
                res = term1 @ U.T
                noise = res * RA_
            else:
                term1 = x * SA_
                res = term1 @ U.T
                noise = res * RA_

            sgn = float(st.sigma_sign)
            eps_eff = sgn * float(sigma)
            return output + eps_eff * noise
        return hook

    def make_lora_flipout_hook(st, sigma, lora_module):
        """Forward hook for loralib.Linear: injects flipout perturbation for both A and B.
        Handles 2D (bs, din), 3D (bs, seq, din), and flattened 2D (bs*seq, din) inputs.
        """
        def hook(module, inputs, output):
            x = inputs[0]
            eps_eff = float(st.sigma_sign) * float(sigma) * float(module.scaling)

            SA_ = st.SA.to(device=x.device, dtype=x.dtype)   # (bs, din)
            RA_ = st.RA.to(device=x.device, dtype=x.dtype)   # (bs, r)
            UA_ = st.UA.to(device=x.device, dtype=x.dtype)   # (r, din)
            SB_ = st.SB.to(device=x.device, dtype=x.dtype)   # (bs, r)
            RB_ = st.RB.to(device=x.device, dtype=x.dtype)   # (bs, dout)
            UB_ = st.UB.to(device=x.device, dtype=x.dtype)   # (dout, r)
            lora_A = module.lora_A.detach().to(dtype=x.dtype)  # (r, din)
            lora_B = module.lora_B.detach().to(dtype=x.dtype)  # (dout, r)

            bs_state = st.RA.shape[0]

            if x.dim() > 2:
                # 3D input: (bs, seq, din) -- transformer attention projections
                SA3 = SA_.unsqueeze(1)   # (bs, 1, din)
                RA3 = RA_.unsqueeze(1)   # (bs, 1, r)
                SB3 = SB_.unsqueeze(1)   # (bs, 1, r)
                RB3 = RB_.unsqueeze(1)   # (bs, 1, dout)
                noise_A = RA3 * ((x * SA3) @ UA_.T) @ lora_B.T   # (bs, seq, dout)
                noise_B = RB3 * ((x @ lora_A.T * SB3) @ UB_.T)   # (bs, seq, dout)
            elif x.dim() == 2 and x.shape[0] != bs_state and x.shape[0] % bs_state == 0:
                # Flattened 2D: (bs*seq, din)
                seq_len = x.shape[0] // bs_state
                x3 = x.view(bs_state, seq_len, -1)
                SA3 = SA_.unsqueeze(1)
                RA3 = RA_.unsqueeze(1)
                SB3 = SB_.unsqueeze(1)
                RB3 = RB_.unsqueeze(1)
                noise_A = (RA3 * ((x3 * SA3) @ UA_.T) @ lora_B.T).view(x.shape[0], -1)
                noise_B = (RB3 * ((x3 @ lora_A.T * SB3) @ UB_.T)).view(x.shape[0], -1)
            else:
                # Standard 2D: (bs, din)
                noise_A = RA_ * ((x * SA_) @ UA_.T) @ lora_B.T   # (bs, dout)
                noise_B = RB_ * ((x @ lora_A.T * SB_) @ UB_.T)   # (bs, dout)

            return output + eps_eff * (noise_A + noise_B)
        return hook

    def enable_flip(sign):
        for m in lora_modules:
            st = states[m]
            st.sigma_sign = sign
            if isinstance(st, PrefixParamState):
                # Direct parameter perturbation: no forward hook needed
                if st.z_keys is not None:
                    orig = m.prefix_keys.data.clone()
                    prefix_perturbed.append((m, 'keys', orig))
                    m.prefix_keys.data = (m.prefix_keys.data.float() + sign * sigma * st.z_keys.float()).to(m.prefix_keys.data.dtype)
                if st.z_values is not None:
                    orig = m.prefix_values.data.clone()
                    prefix_perturbed.append((m, 'values', orig))
                    m.prefix_values.data = (m.prefix_values.data.float() + sign * sigma * st.z_values.float()).to(m.prefix_values.data.dtype)
            elif is_lora:
                h = m.register_forward_hook(make_lora_flipout_hook(st, sigma, m))
                hooks.append(h)
            else:
                # Register appropriate hook based on module type
                if isinstance(m, nn.Embedding):
                    h = m.register_forward_hook(make_embedding_hook_simple(st, sigma))
                elif isinstance(m, (nn.LayerNorm,)) or type(m).__name__ in ['LlamaRMSNorm', 'RMSNorm']:
                    h = m.register_forward_hook(make_norm_hook_simple(st, sigma))
                else:
                    h = m.register_forward_hook(get_flipout_hook(st, sigma))
                hooks.append(h)

    def disable_flip():
        # Restore directly-perturbed prefix parameters
        for m, which, orig in prefix_perturbed:
            if which == 'keys':
                m.prefix_keys.data = orig
            else:
                m.prefix_values.data = orig
        prefix_perturbed.clear()
        for h in hooks:
            h.remove()
        hooks.clear()

    # 2) Estimation: perturb in one or both directions based on estimation_side
    enable_flip(+1)
    loss_plus = compute_per_example_loss(model, inputs)    # (bs,) float32
    disable_flip()

    if estimation_side == "one":
        # Single-sided estimation (like original GRZO with variance reduction)
        assert loss_plus.shape[0] in (bs_flat, bs_eff), \
            f"loss_plus shape {loss_plus.shape[0]} not in (bs_flat={bs_flat}, bs_eff={bs_eff})"

        # Single-sided gradient estimation with mean subtraction
        losses = loss_plus.float()  # (bs_eff,) or (bs_flat,)
        mu_L = losses.mean()
        sigma_L = losses.std()
        if adv_std_floor > 0:
            sigma_L = torch.clamp(sigma_L, min=adv_std_floor)
        adv = (losses - mu_L) / (sigma_L + eps)  # Variance reduction
        divisor = sigma  # Single-sided: divide by sigma

        return_loss = loss_plus.mean().item()
    elif estimation_side == "uniform":
        # Uniform (unweighted) averaging -- Flipout directions without GR weighting (ablation)
        assert loss_plus.shape[0] in (bs_flat, bs_eff), \
            f"loss_plus shape {loss_plus.shape[0]} not in (bs_flat={bs_flat}, bs_eff={bs_eff})"

        adv = torch.ones_like(loss_plus.float())
        divisor = sigma

        return_loss = loss_plus.mean().item()
    elif estimation_side in ("two", "two_norm"):
        # Two-sided estimation
        enable_flip(-1)
        loss_minus = compute_per_example_loss(model, inputs)   # (bs,) float32
        disable_flip()

        assert loss_plus.shape[0] in (bs_flat, bs_eff), \
            f"loss_plus shape {loss_plus.shape[0]} not in (bs_flat={bs_flat}, bs_eff={bs_eff})"
        assert loss_minus.shape[0] in (bs_flat, bs_eff), \
            f"loss_minus shape {loss_minus.shape[0]} not in (bs_flat={bs_flat}, bs_eff={bs_eff})"

        delta = (loss_plus.float() - loss_minus.float()) / 2.0
        if estimation_side == "two_norm":
            # Two-sided with std normalization: delta / (std(delta) + eps)
            # No mean subtraction since two-sided differences already have ~zero mean
            sigma_delta = delta.std()
            if adv_std_floor > 0:
                sigma_delta = torch.clamp(sigma_delta, min=adv_std_floor)
            adv = delta / (sigma_delta + eps)
        else:
            adv = delta  # No variance reduction
        divisor = 2.0 * sigma  # Two-sided: divide by 2*sigma

        return_loss = (loss_plus.mean().item() + loss_minus.mean().item()) / 2.0

    if adv_clip > 0:
        adv = adv.clamp(min=-adv_clip, max=adv_clip)
    
    # 3) GRZO gradient estimation
    # Note: For multi-choice tasks, loss has shape (bs_eff,) where bs_eff = number of questions
    # Each loss value reflects model performance across all options for that question
    
    # Create expanded advantage vector for flattened multi-choice batches
    # For multi-choice: bs_flat = total options (e.g., 44), bs_eff = questions (e.g., 7)
    # Each question's advantage is replicated to all its options
    # This is correct because: R/S are per-option perturbations, but advantage weights
    # should be per-question (all options of a question share the same loss signal)
    if loss_plus.shape[0] == bs_eff and bs_flat != bs_eff and "num_options" in inputs and inputs["num_options"] is not None:
        # loss_plus is (bs_eff,): need to expand to (bs_flat,)
        # Map each question's advantage to all its options
        # Example: question 0 has 4 options -> adv[0] is used for options 0-3
        adv_list = []
        for i, n_opt in enumerate(inputs["num_options"]):
            adv_list.extend([adv[i].item()] * n_opt)
        adv_expanded = torch.tensor(adv_list, dtype=adv.dtype, device=adv.device)  # (bs_flat,)
    else:
        # loss_plus is already (bs_flat,): no expansion needed
        adv_expanded = adv
    
    # Scale adv by lr/sigma here or later?
    # Update rule: W <- W - lr * Grad
    # Grad_W ~ (U * MA) / (2 * sigma * N) ?? 
    # Just use similar scaling as LoRA.
    # grad_A = (st.UA.float() * MA) / (2.0 * sigma * bs_eff)
    
    # 4) closed-form grad for each LoRA A/B using flipout signs
    # Handle tied weights (e.g., embed_tokens.weight == lm_head.weight in many LLMs)
    # Track updated weights by data_ptr to avoid double-updating shared parameters
    updated_weights = set()
    
    for m in lora_modules:
        st = states[m]

        if isinstance(st, PrefixParamState):
            # Prefix parameter update: grad = mean(adv) / divisor * z
            # (shared noise across batch, so gradient is scalar-weighted)
            scalar_adv = adv_expanded.float().mean()
            if st.z_keys is not None and hasattr(m, 'prefix_keys'):
                ptr = m.prefix_keys.data.data_ptr()
                if ptr not in updated_weights:
                    grad = (scalar_adv / divisor) * st.z_keys.float()
                    m.prefix_keys.data = (m.prefix_keys.data.float() - (lr / total_accum_steps) * grad).to(m.prefix_keys.data.dtype)
                    updated_weights.add(ptr)
            if st.z_values is not None and hasattr(m, 'prefix_values'):
                ptr = m.prefix_values.data.data_ptr()
                if ptr not in updated_weights:
                    grad = (scalar_adv / divisor) * st.z_values.float()
                    m.prefix_values.data = (m.prefix_values.data.float() - (lr / total_accum_steps) * grad).to(m.prefix_values.data.dtype)
                    updated_weights.add(ptr)
            continue

        # Skip layers that ended up on meta device (device_map='auto' under memory pressure)
        _ref = m.weight if hasattr(m, 'weight') and m.weight is not None else \
               (m.lora_A.data if hasattr(m, 'lora_A') else None)
        if _ref is not None and _ref.device.type == 'meta':
            continue

        if is_lora:
            # Check if lora_A already updated (tied weights)
            weight_ptr_A = m.lora_A.data.data_ptr()
            weight_ptr_B = m.lora_B.data.data_ptr()
            if weight_ptr_A in updated_weights or weight_ptr_B in updated_weights:
                continue
            
            # A grad
            RA = st.RA.float()      # (bs_flat, r)
            SA = st.SA.float()      # (bs_flat, din)
            MA = RA.t() @ (adv_expanded[:, None] * SA)        # (r, din)
            grad_A = (st.UA.float() * MA) / (divisor * bs_eff)

            # B grad
            RB = st.RB.float()      # (bs_flat, dout)
            SB = st.SB.float()      # (bs_flat, r)
            MB = RB.t() @ (adv_expanded[:, None] * SB)        # (dout, r)
            grad_B = (st.UB.float() * MB) / (divisor * bs_eff)

            # Immediate partial update: mathematically equivalent to accumulate-then-apply
            # for ZO since each step's estimate is independent and unbiased.
            m.lora_A.data = (m.lora_A.data.float() - (lr / total_accum_steps) * grad_A).to(m.lora_A.data.dtype)
            m.lora_B.data = (m.lora_B.data.float() - (lr / total_accum_steps) * grad_B).to(m.lora_B.data.dtype)
            
            # Mark as updated
            updated_weights.add(weight_ptr_A)
            updated_weights.add(weight_ptr_B)
        else:
            # Full FT Gradient - handle different module types
            
            # Check if weight already updated (tied weights, e.g., embed_tokens == lm_head)
            weight_ptr = m.weight.data.data_ptr()
            if weight_ptr in updated_weights:
                continue
            
            if isinstance(m, nn.Embedding):
                # Embedding: simplified sparse row update using cached uniq/inv/ex_ids from forward
                if st.last_uniq is None:
                    continue  # Hook wasn't called, skip

                uniq = st.last_uniq
                inv = st.last_inv
                ex_ids = st.last_ex_ids
                device = m.weight.device
                D = m.weight.shape[1]

                # occurrence scalar signs
                ids_flat = st.last_ids.reshape(-1).to(torch.int64)
                s_occ = _rademacher_scalar(ex_ids, ids_flat, st.seed_s, device)  # (Nocc,) float32

                # coeff per occurrence
                a_occ = adv_expanded.to(torch.float32)[ex_ids]  # (Nocc,)
                coeff_occ = a_occ * s_occ                       # (Nocc,)

                # reduce to per-row scalar coeff: c_t
                K = uniq.numel()
                c_rows = torch.zeros((K,), device=device, dtype=torch.float32)
                c_rows.index_add_(0, inv, coeff_occ)            # (K,)

                # row direction vectors u_t
                u_rows = _rademacher_rows_dims(uniq, D, st.seed_u, device)  # (K,D) float32

                grad_rows = (c_rows[:, None] * u_rows) / (divisor * bs_eff)  # (K,D)

                # Immediate partial sparse update
                m.weight.data.index_add_(0, uniq, (-(lr / total_accum_steps) * grad_rows).to(m.weight.dtype))
                
                # Mark as updated
                updated_weights.add(weight_ptr)
                
            elif isinstance(m, (nn.LayerNorm,)) or type(m).__name__ in ['LlamaRMSNorm', 'RMSNorm']:
                # LayerNorm/RMSNorm: simplified update using per-example independent +/-1 vectors
                device = m.weight.device
                D = m.weight.shape[0]
                B = st.bs

                u = _rademacher_2d(B, D, st.seed, device).to(torch.float32)  # (B,D)
                a = adv_expanded.to(torch.float32).view(B, 1)                 # (B,1)

                grad_w = (a * u).sum(dim=0) / (divisor * bs_eff)  # (D,)

                # Immediate partial update
                m.weight.data = (m.weight.data.float() - (lr / total_accum_steps) * grad_w).to(m.weight.data.dtype)

                # bias: same u (can use different seed if needed)
                if hasattr(m, "bias") and m.bias is not None:
                    grad_b = (a * u).sum(dim=0) / (divisor * bs_eff)
                    # Immediate partial update
                    m.bias.data = (m.bias.data.float() - (lr / total_accum_steps) * grad_b).to(m.bias.data.dtype)
                
                del u, grad_w
                
                # Mark as updated
                updated_weights.add(weight_ptr)
                
            else:
                # Linear layer: weight shape (Dout, Din)
                # Generate U for gradient estimation
                # IMPORTANT: Use m.weight.dtype (fp16/bf16) to match forward pass, then .float() for computation
                if lozo_uv is not None and m in lozo_uv:
                    # GRZO + LOZO: same "u @ v / sqrt(r)" as the forward hook.
                    _u, _v = lozo_uv[m]
                    _r_norm = float(_u.shape[1]) ** 0.5
                    U = (_u.to(device=m.weight.device, dtype=torch.float32)
                         @ _v.to(device=m.weight.device, dtype=torch.float32)) / _r_norm
                else:
                    rng = torch.Generator(device=m.weight.device)
                    rng.manual_seed(st.seed)
                    U = generate_noise(m.weight.shape, rng, m.weight.dtype, m.weight.device, st.u_distribution).float()

                # GRZO + QuZO: gradient copy u_{i,2} (independent stochastic
                # rounding seed from the forward copy, so E[u_1 u_2^T] = U U^T).
                if quant_bits is not None:
                    _q_rng = torch.Generator(device=U.device)
                    _q_rng.manual_seed(int(st.seed) ^ 0xC0FFEE5)
                    U = _zo_quant_dequant_sym(U, quant_bits, generator=_q_rng)

                # GRZO + SparseMeZO: same mask used in the forward hook is
                # applied to the gradient so unmasked weights stay frozen.
                if sparse_masks is not None and m in sparse_masks:
                    U.mul_(sparse_masks[m].to(device=U.device, dtype=U.dtype))

                # DeltaW = U * (R @ S.T) element wise
                # R: (B, Dout), S: (B, Din)
                RA = st.RA.float()
                SA = st.SA.float()

                # MA = (R_weighted).T @ S --> (Dout, B) @ (B, Din) --> (Dout, Din)
                MA = (adv_expanded[:, None] * RA).t() @ SA
                del RA, SA  # free immediately; only MA is needed below

                # Compute grad_W in-place to avoid a 3rd large allocation:
                # U *= MA  (elementwise, in-place), then scale; U now holds grad_W
                U.mul_(MA)
                del MA
                U.div_(divisor * bs_eff)
                # Immediate partial update (no storage dict needed)
                U.mul_(lr / total_accum_steps)
                m.weight.data = (m.weight.data.float() - U).to(m.weight.data.dtype)
                del U

                # Mark as updated
                updated_weights.add(weight_ptr)

    # return scalar logging loss
    return float(return_loss)


# ===========================================================================
# DDP-correct version of grzo_flipout_step.
#
# Design summary
# --------------
#   * "master_seed" is broadcast from rank 0. After broadcast all ranks call
#     "torch.manual_seed(master_seed)" (Phase 1). Under that seed they build
#     "FlipoutState" / "NormNoiseState" / "EmbNoiseState", which makes the
#     state seeds for the per-weight noise "U" (and the row direction "u_t"
#     for Embedding) identical on every rank.
#   * Phase 2 reseeds with "master_seed + 10007*(rank+1)" and regenerates the
#     per-example Rademacher signs R/S (Linear). For NormNoiseState we just
#     perturb "st.seed" so the per-example "u_i" differs per rank. For
#     EmbNoiseState the hook uses *global* "ex_ids = rank*B + arange(B)" so
#     the per-occurrence sign "s_occ" is naturally per-rank.
#   * Per-example losses are "all_gather"ed -> all ranks see the global loss
#     vector -> all ranks compute the same global advantage.
#   * Linear & Norm gradients are computed locally; the partial response
#     matrix "MA_local" is "all_reduce(SUM)"ed -> all ranks have the global
#     "MA_global". Then each rank computes "grad = U * MA_global / (div *
#     bs_eff_global)" and applies it in-place; since "U" is shared all ranks
#     apply the same update -> weights stay in sync without DDP gradient hooks.
#   * Embedding: each rank has its local "(uniq, c_rows_local)". We use
#     "dist.all_gather_object" to share these across ranks and merge them.
#     All ranks then apply the same sparse update.
#
# Scope of the MVP
# ----------------
#   * Full FT only (no LoRA, no prefix tuning).
#   * estimation_side in {"two", "two_norm"} (one-sided/uniform omitted for
#     now). The default sweep target uses "two".
#   * "num_options" may vary across ranks ("bs_eff_local", "bs_flat_local"
#     are obtained per rank; the global counts are gathered).
#   * "accumulate_grad" is not supported in this MVP.
# ===========================================================================

@torch.no_grad()
def grzo_flipout_step_ddp(
    model, inputs, lora_modules, sigma, lr, eps=1e-8,
    u_distribution="gaussian", estimation_side="two",
    accumulate_grad=False, grad_accumulation_storage=None,
    accum_step=1, total_accum_steps=1,
    sparse_masks=None, lozo_uv=None, quant_bits=None,
    adv_std_floor: float = 0.0, adv_clip: float = 0.0,
    lozo_strict: bool = False,
):
    """DDP-correct version of grzo_flipout_step.

    Mathematically equivalent to running the single-GPU "grzo_flipout_step"
    with global batch = (world_size * per_device_batch_size) and the same
    master seed.

    sparse_masks / lozo_uv: same semantics as the single-GPU version (see
    grzo_flipout_step docstring). Masks / (u,v) factors must be deterministic
    on every rank — typically derived from the model weights (sparse) or from
    the master-broadcast seed (lozo).
    """
    assert dist is not None and dist.is_initialized() and dist.get_world_size() > 1, \
        "grzo_flipout_step_ddp called outside a distributed group"
    if accumulate_grad:
        raise NotImplementedError("DDP flipout does not yet support gradient accumulation")
    if estimation_side not in ("two", "two_norm"):
        raise NotImplementedError(
            f"DDP flipout currently only supports estimation_side in {{'two','two_norm'}}, got {estimation_side}"
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    bs_flat_local = inputs["input_ids"].shape[0]
    if "num_options" in inputs and inputs["num_options"] is not None:
        no_local = inputs["num_options"]
        if hasattr(no_local, "tolist"):
            no_local_list = no_local.tolist()
        else:
            no_local_list = list(no_local)
        bs_eff_local = len(no_local_list)
    else:
        no_local_list = [1] * bs_flat_local
        bs_eff_local = bs_flat_local

    # ---- 1. Master seed broadcast (rank 0 -> all) ----
    master_seed_t = torch.empty(1, device=device, dtype=torch.int64)
    if rank == 0:
        master_seed_t.fill_(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    dist.broadcast(master_seed_t, src=0)
    master_seed = int(master_seed_t.item())

    # ---- 2. Detect supported module configuration ----
    is_lora = False
    if len(lora_modules) > 0 and hasattr(lora_modules[0], "lora_A"):
        is_lora = True
    if is_lora:
        raise NotImplementedError("DDP flipout MVP: LoRA path not supported yet")
    for m in lora_modules:
        if hasattr(m, "prefix_keys") or hasattr(m, "prefix_values"):
            raise NotImplementedError("DDP flipout MVP: prefix-tuning path not supported yet")

    # ---- 3. Phase 1: master-seeded state construction (U-seeds shared across ranks) ----
    torch.manual_seed(master_seed)
    torch.cuda.manual_seed_all(master_seed)

    states = {}
    for m in lora_modules:
        if isinstance(m, nn.Embedding):
            states[m] = EmbNoiseState(bs=bs_flat_local)
        elif isinstance(m, nn.LayerNorm) or type(m).__name__ in ("LlamaRMSNorm", "RMSNorm"):
            dim = m.weight.shape[0]
            states[m] = NormNoiseState(bs=bs_flat_local, dim=dim)
        else:
            # Plain Linear (full FT).
            shape = tuple(m.weight.shape)
            states[m] = FlipoutState(
                bs_flat_local, shape, device=device, dtype=dtype,
                is_lora=False, u_distribution=u_distribution,
            )

    # ---- 4. Phase 2: rank-specific reseeding -> regenerate per-example R/S ----
    rank_seed = (master_seed + 10007 * (rank + 1)) % (2**31)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    for m, st in states.items():
        if isinstance(st, FlipoutState):
            st.RA = (torch.randint(0, 2, st.RA.shape, device=st.RA.device, dtype=st.RA.dtype) * 2 - 1)
            st.SA = (torch.randint(0, 2, st.SA.shape, device=st.SA.device, dtype=st.SA.dtype) * 2 - 1)
            # Strict LOZO-GRZO: drop per-example column sign s_i so the update
            # stays in span(V). After Phase 2 reseed, override SA to ones.
            if lozo_strict and lozo_uv is not None and m in lozo_uv:
                st.SA = torch.ones_like(st.SA)
        elif isinstance(st, NormNoiseState):
            # The hook reads "_rademacher_2d(B, D, st.seed, device)" for the per-example
            # +/-1 vector. Perturb the seed so different ranks get different u_i.
            st.seed = (st.seed + 1000003 * (rank + 1)) % (2**31)
        elif isinstance(st, EmbNoiseState):
            # seed_u (row direction) stays shared so the same token row gets the
            # same direction across ranks. seed_s is shared too -- per-occurrence
            # divergence comes from the rank-shifted ex_ids in the hook below.
            pass

    rank_offset = rank * bs_flat_local  # global offset into the [0..bs_flat_global) example index space

    # ---- 5. Hook factories (Norm & Linear identical to single-GPU; Embedding rank-aware) ----
    hooks = []

    def make_embedding_hook_ddp(st: EmbNoiseState, sigma_: float, rank_off: int):
        def hook(module, _inputs, output):
            if module.weight.device.type == "meta":
                return output
            ids = _inputs[0]
            out = output
            squeeze_back = False
            if ids.dim() == 1:
                ids = ids.unsqueeze(1)
                out = out.unsqueeze(1)
                squeeze_back = True
            B, S, D = out.shape
            dev = ids.device
            assert B == st.bs, f"Embedding batch mismatch: got B={B}, expected {st.bs}"
            ids_flat = ids.reshape(-1).to(torch.int64)
            # Rank-shifted global example ids so that per-occurrence signs differ across ranks.
            ex_ids = (
                torch.arange(rank_off, rank_off + B, device=dev, dtype=torch.int64)
                .view(B, 1).expand(B, S).reshape(-1)
            )
            uniq, inv = torch.unique(ids_flat, sorted=True, return_inverse=True)
            u_rows = _rademacher_rows_dims(uniq, D, st.seed_u, dev).to(out.dtype)
            s_occ = _rademacher_scalar(ex_ids, ids_flat, st.seed_s, dev).to(out.dtype)
            noise = u_rows[inv] * s_occ[:, None]
            eps_eff = float(st.sigma_sign) * float(sigma_)
            out = out + eps_eff * noise.view(B, S, D)
            st.last_ids = ids.detach()
            st.last_uniq = uniq
            st.last_inv = inv
            st.last_ex_ids = ex_ids
            if squeeze_back:
                out = out.squeeze(1)
            return out
        return hook

    def make_norm_hook(st: NormNoiseState, sigma_: float):
        def hook(module, _inputs, output):
            if module.weight.device.type == "meta":
                return output
            x = _inputs[0]
            D = x.shape[-1]
            bs_state = st.bs
            out_dtype = output.dtype
            dev = x.device
            eps_eff = float(st.sigma_sign) * float(sigma_)
            if x.shape[0] == bs_state:
                B = bs_state
                xhat = _normed_x(module, x).to(out_dtype)
                u = _rademacher_2d(B, D, st.seed, dev).to(out_dtype)
                while u.dim() < xhat.dim():
                    u = u.unsqueeze(1)
                return output + eps_eff * (xhat * u)
            if x.dim() == 2 and x.shape[0] % bs_state == 0:
                S = x.shape[0] // bs_state
                x2 = x.view(bs_state, S, D)
                out2 = output.view(bs_state, S, D)
                xhat = _normed_x(module, x2).to(out_dtype)
                u = _rademacher_2d(bs_state, D, st.seed, dev).to(out_dtype).view(bs_state, 1, D)
                out2 = out2 + eps_eff * (xhat * u)
                return out2.view(x.shape[0], D)
            return output
        return hook

    def make_linear_hook(st: FlipoutState, sigma_: float):
        def hook(module, _inputs, output):
            if module.weight.device.type == "meta":
                return output
            x = _inputs[0]
            if lozo_uv is not None and module in lozo_uv:
                _u, _v = lozo_uv[module]
                _r_norm = float(_u.shape[1]) ** 0.5  # rank
                U = (_u.to(device=module.weight.device, dtype=module.weight.dtype)
                     @ _v.to(device=module.weight.device, dtype=module.weight.dtype)) / _r_norm
            else:
                rng = torch.Generator(device=module.weight.device)
                rng.manual_seed(st.seed)
                U = generate_noise(module.weight.shape, rng, module.weight.dtype, module.weight.device, st.u_distribution)
            if quant_bits is not None:
                # QuZO u_{i,1}: forward copy, independent stochastic rounding seed
                _q_rng = torch.Generator(device=U.device)
                _q_rng.manual_seed(int(st.seed) ^ 0xF0B1A11)
                U = _zo_quant_dequant_sym(U, quant_bits, generator=_q_rng)
            if sparse_masks is not None and module in sparse_masks:
                U = U * sparse_masks[module].to(device=U.device, dtype=U.dtype)
            bs_state = st.RA.shape[0]
            SA_ = st.SA.to(device=x.device, dtype=x.dtype)
            RA_ = st.RA.to(device=x.device, dtype=x.dtype)
            if x.dim() == 2 and x.shape[0] != bs_state and x.shape[0] % bs_state == 0:
                seq_len = x.shape[0] // bs_state
                x_re = x.view(bs_state, seq_len, -1)
                SA_ = SA_.view(bs_state, 1, -1)
                RA_ = RA_.view(bs_state, 1, -1)
                noise = ((x_re * SA_) @ U.T) * RA_
                noise = noise.view(x.shape[0], -1)
            elif x.dim() > 2:
                SA_ = SA_.view(bs_state, 1, -1)
                RA_ = RA_.view(bs_state, 1, -1)
                noise = ((x * SA_) @ U.T) * RA_
            else:
                noise = ((x * SA_) @ U.T) * RA_
            eps_eff = float(st.sigma_sign) * float(sigma_)
            return output + eps_eff * noise
        return hook

    def enable_flip(sign):
        for m_ in lora_modules:
            st_ = states[m_]
            st_.sigma_sign = sign
            if isinstance(m_, nn.Embedding):
                h = m_.register_forward_hook(make_embedding_hook_ddp(st_, sigma, rank_offset))
            elif isinstance(m_, nn.LayerNorm) or type(m_).__name__ in ("LlamaRMSNorm", "RMSNorm"):
                h = m_.register_forward_hook(make_norm_hook(st_, sigma))
            else:
                h = m_.register_forward_hook(make_linear_hook(st_, sigma))
            hooks.append(h)

    def disable_flip():
        for h in hooks:
            h.remove()
        hooks.clear()

    # ---- 6. Two forward passes (theta+sigma, theta-sigma) ----
    enable_flip(+1)
    loss_plus_local = compute_per_example_loss(model, inputs).float()  # (bs_flat_local,) or (bs_eff_local,)
    disable_flip()

    enable_flip(-1)
    loss_minus_local = compute_per_example_loss(model, inputs).float()
    disable_flip()

    # Local loss vectors might have shape (bs_flat_local,) (per-option) or
    # (bs_eff_local,) (per-question for multi-choice). Normalize to (bs_eff_local,):
    if loss_plus_local.shape[0] == bs_flat_local and bs_flat_local != bs_eff_local:
        # Aggregate per-option to per-question by group mean.
        agg = torch.zeros(bs_eff_local, device=loss_plus_local.device, dtype=loss_plus_local.dtype)
        idx = 0
        for i, n in enumerate(no_local_list):
            agg[i] = loss_plus_local[idx:idx + int(n)].mean()
            idx += int(n)
        loss_plus_local = agg
        agg = torch.zeros(bs_eff_local, device=loss_minus_local.device, dtype=loss_minus_local.dtype)
        idx = 0
        for i, n in enumerate(no_local_list):
            agg[i] = loss_minus_local[idx:idx + int(n)].mean()
            idx += int(n)
        loss_minus_local = agg
    # Now loss_*_local is (bs_eff_local,)

    # ---- 7. All-gather per-example losses to compute global advantage ----
    # bs_eff_local must be the same across all ranks for all_gather_into_tensor.
    # For SuperGLUE classification tasks (RTE/SST2/CB/BoolQ/COPA/WiC) num_options is fixed,
    # so bs_eff_local is constant. (Variable-num_options multi-choice is out-of-scope for MVP.)
    loss_plus_global = torch.empty(world_size * bs_eff_local, device=device, dtype=loss_plus_local.dtype)
    loss_minus_global = torch.empty(world_size * bs_eff_local, device=device, dtype=loss_minus_local.dtype)
    dist.all_gather_into_tensor(loss_plus_global, loss_plus_local.contiguous())
    dist.all_gather_into_tensor(loss_minus_global, loss_minus_local.contiguous())

    bs_eff_global = world_size * bs_eff_local
    bs_flat_global = world_size * bs_flat_local

    # ---- 8. Global advantage ----
    delta = (loss_plus_global - loss_minus_global) / 2.0
    if estimation_side == "two_norm":
        sigma_delta = delta.std()
        adv_global = delta / (sigma_delta + eps)
    else:
        adv_global = delta
    divisor = 2.0 * sigma
    return_loss = (loss_plus_global.mean().item() + loss_minus_global.mean().item()) / 2.0

    # Expand per-question adv to per-option if needed (using local num_options list of each rank gathered globally).
    if bs_flat_global != bs_eff_global:
        all_no = [None] * world_size
        dist.all_gather_object(all_no, no_local_list)
        flat_no = [n for lst in all_no for n in lst]  # len = bs_eff_global
        adv_list = []
        for i, n in enumerate(flat_no):
            adv_list.extend([adv_global[i].item()] * int(n))
        adv_global_flat = torch.tensor(adv_list, dtype=adv_global.dtype, device=adv_global.device)
    else:
        adv_global_flat = adv_global

    # Slice for local examples (in the flat space).
    adv_local_flat = adv_global_flat[rank_offset:rank_offset + bs_flat_local]

    # ---- 9. Gradient computation + apply update (per-module) ----
    # Multi-node DDP strategy: instead of all_reduce'ing the per-module gradient
    # matrices (each MA is dout*din floats = up to 224 MB per Linear, 30 GB total
    # for Llama-3-8B which both saturates memory and blows latency budget when
    # done as 200+ separate NCCL calls), we all_gather the per-example signs
    # (R, s) across ranks. The signs are KB-sized vectors, so the total transfer
    # drops by ~250x and each rank locally reconstructs the identical global
    # MA from the gathered signs.
    #
    # Math equivalence:
    #   MA_global = Σ_r (adv_r ⊙ R_r)ᵀ @ s_r          (current all_reduce)
    #             = (adv_global ⊙ R_global)ᵀ @ s_global  (with concat'd globals)
    # both produce identical (dout, din) matrices.
    updated_weights = set()

    # Pass 1: per-module work — Embedding handled inline (uses sparse rows
    # which can't be batched cleanly via all_gather_into_tensor). Linear and
    # LN have R/s/u tensors collected into a flat all_gather batch.
    pending_linear = []   # (m, R_local, S_local)        — fp32
    pending_ln = []       # (m, u_local)                 — fp32 (bs_local, D)
    for m in lora_modules:
        st = states[m]
        _ref = m.weight if hasattr(m, "weight") and m.weight is not None else None
        if _ref is None or _ref.device.type == "meta":
            continue
        weight_ptr = m.weight.data.data_ptr()
        if weight_ptr in updated_weights:
            continue

        if isinstance(m, nn.Embedding):
            # Embedding: inline (sparse, uses all_gather_object).
            if st.last_uniq is None:
                continue
            uniq_local = st.last_uniq
            inv = st.last_inv
            ex_ids = st.last_ex_ids
            dev = m.weight.device
            D = m.weight.shape[1]
            ids_flat = st.last_ids.reshape(-1).to(torch.int64)
            s_occ = _rademacher_scalar(ex_ids, ids_flat, st.seed_s, dev)
            a_occ = adv_global_flat.to(torch.float32)[ex_ids]
            coeff_occ = a_occ * s_occ
            K_local = uniq_local.numel()
            c_rows_local = torch.zeros((K_local,), device=dev, dtype=torch.float32)
            c_rows_local.index_add_(0, inv, coeff_occ)
            gathered = [None] * world_size
            dist.all_gather_object(
                gathered,
                (uniq_local.detach().cpu().tolist(), c_rows_local.detach().cpu().tolist()),
            )
            merged = {}
            for uniq_r, c_r in gathered:
                for u_, c_ in zip(uniq_r, c_r):
                    merged[int(u_)] = merged.get(int(u_), 0.0) + float(c_)
            if not merged:
                continue
            uniq_global_list = sorted(merged.keys())
            uniq_global = torch.tensor(uniq_global_list, device=dev, dtype=torch.int64)
            c_rows_global = torch.tensor(
                [merged[u_] for u_ in uniq_global_list], device=dev, dtype=torch.float32,
            )
            u_rows = _rademacher_rows_dims(uniq_global, D, st.seed_u, dev)
            grad_rows = (c_rows_global[:, None] * u_rows) / (divisor * bs_eff_global)
            m.weight.data.index_add_(
                0, uniq_global,
                (-(lr / total_accum_steps) * grad_rows).to(m.weight.dtype),
            )
            updated_weights.add(weight_ptr)

        elif isinstance(m, nn.LayerNorm) or type(m).__name__ in ("LlamaRMSNorm", "RMSNorm"):
            dev = m.weight.device
            D = m.weight.shape[0]
            B_local = st.bs
            u_local = _rademacher_2d(B_local, D, st.seed, dev).to(torch.float32)
            pending_ln.append((m, u_local))

        else:
            # Linear: just collect R, S (signs); MA recomputed in pass 2 locally
            pending_linear.append((m, st.RA.float(), st.SA.float()))

    # Batched all_gather of all per-rank R, S, u tensors into globals.
    # All ranks see the same global signs after this point.
    gathered_R = {}   # m -> R_global  (bs_global, dout)  fp32
    gathered_S = {}   # m -> S_global  (bs_global, din)   fp32
    gathered_U = {}   # m -> u_global  (bs_global, D)     fp32  (for LN)
    if pending_linear or pending_ln:
        with dist._coalescing_manager():
            for m, R_local, S_local in pending_linear:
                bs, dout = R_local.shape
                _, din = S_local.shape
                R_global = torch.empty(bs * world_size, dout, device=R_local.device, dtype=R_local.dtype)
                S_global = torch.empty(bs * world_size, din, device=S_local.device, dtype=S_local.dtype)
                dist.all_gather_into_tensor(R_global, R_local.contiguous())
                dist.all_gather_into_tensor(S_global, S_local.contiguous())
                gathered_R[m] = R_global
                gathered_S[m] = S_global
            for m, u_local in pending_ln:
                bs, D = u_local.shape
                u_global = torch.empty(bs * world_size, D, device=u_local.device, dtype=u_local.dtype)
                dist.all_gather_into_tensor(u_global, u_local.contiguous())
                gathered_U[m] = u_global

    # Free the local R/S/u; we have the globals now
    del pending_linear, pending_ln

    # Pass 2: each module computes its global gradient locally and applies it.
    # adv_global_flat already exists (derived from all_gathered losses earlier).
    adv_g = adv_global_flat.to(torch.float32)

    for m, R_g in gathered_R.items():
        st = states[m]
        weight_ptr = m.weight.data.data_ptr()
        if weight_ptr in updated_weights:
            continue
        S_g = gathered_S[m]
        # MA_global = (adv_g ⊙ R_g)ᵀ @ S_g    shape (dout, din)
        MA_global = (adv_g[:, None] * R_g).t() @ S_g
        del R_g, S_g

        # Generate U (same on all ranks; uses master-seeded RNG or LOZO uv).
        if lozo_uv is not None and m in lozo_uv:
            _u, _v = lozo_uv[m]
            _r_norm = float(_u.shape[1]) ** 0.5
            U = (_u.to(device=m.weight.device, dtype=torch.float32)
                 @ _v.to(device=m.weight.device, dtype=torch.float32)) / _r_norm
        else:
            rng = torch.Generator(device=m.weight.device)
            rng.manual_seed(st.seed)
            U = generate_noise(m.weight.shape, rng, m.weight.dtype, m.weight.device, st.u_distribution).float()
        if quant_bits is not None:
            # QuZO u_{i,2}: gradient copy, independent stochastic rounding seed
            _q_rng = torch.Generator(device=U.device)
            _q_rng.manual_seed(int(st.seed) ^ 0xC0FFEE5)
            U = _zo_quant_dequant_sym(U, quant_bits, generator=_q_rng)
        if sparse_masks is not None and m in sparse_masks:
            U.mul_(sparse_masks[m].to(device=U.device, dtype=U.dtype))
        U.mul_(MA_global)
        del MA_global
        U.div_(divisor * bs_eff_global)
        U.mul_(lr / total_accum_steps)
        m.weight.data = (m.weight.data.float() - U).to(m.weight.data.dtype)
        del U
        updated_weights.add(weight_ptr)

    # LN updates
    for m, u_g in gathered_U.items():
        weight_ptr = m.weight.data.data_ptr()
        if weight_ptr in updated_weights:
            continue
        # grad_w_global = (adv_g * u_g).sum(dim=0)
        grad_w = (adv_g[:, None] * u_g).sum(dim=0) / (divisor * bs_eff_global)
        del u_g
        m.weight.data = (m.weight.data.float() - (lr / total_accum_steps) * grad_w).to(m.weight.data.dtype)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data = (m.bias.data.float() - (lr / total_accum_steps) * grad_w).to(m.bias.data.dtype)
        updated_weights.add(weight_ptr)

    return float(return_loss)


