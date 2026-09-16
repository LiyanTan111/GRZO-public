"""
QuZO weight fake-quantization (Algorithm 1).

Replaces transformer-block nn.Linear with a subclass that fake-quantizes
its weight on every forward pass. Storage stays fp16/fp32 (nn.Parameter),
but compute uses int-W bit precision via per-channel symmetric absmax.

Matches paper external_zo/QuZO/large_models/quant_func/quant_modules.py
LinearQuantizer + quant_model.quantize_model, simplified to remove the
custom CUDA codebook kernel (replaced by pytorch ops) and the MSE alpha
search (replaced by runtime absmax — fine because ZO updates are tiny).

Compose with perturbation-quantization (--quant_bits, set separately) to
get the full QuZO recipe: weight bits = Wbit, perturbation bits = Pbit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightQuantLinear(nn.Linear):
    """nn.Linear with per-channel symmetric absmax fake-quant on weight.

    On forward: w_q = round_clip(w / scale) * scale,
        scale = max|w|_row / (2^(wbit-1) - 1).
    The underlying self.weight Parameter is the fp16 storage; perturbations
    (ZO eps*z) act on it directly. Fake-quant is recomputed each forward.

    Hooks / DDP / state_dict all behave identically to nn.Linear because we
    subclass it without adding any registered buffers/params beyond the base.
    """

    @classmethod
    def from_linear(cls, linear: nn.Linear, wbit: int = 4) -> "WeightQuantLinear":
        out_features, in_features = linear.weight.shape
        has_bias = linear.bias is not None
        m = cls(in_features, out_features, bias=has_bias,
                device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            m.weight.copy_(linear.weight.data)
            if has_bias:
                m.bias.copy_(linear.bias.data)
        m._wbit = int(wbit)
        m._num_levels = 2 ** (wbit - 1) - 1
        return m

    def _ensure_init(self):
        # Tolerate construction without from_linear (e.g. DDP wrap reconstruction).
        if not hasattr(self, "_wbit"):
            self._wbit = 4
            self._num_levels = 2 ** (self._wbit - 1) - 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._ensure_init()
        w = self.weight
        num_levels = self._num_levels
        # Per-output-channel symmetric absmax → scale shape (out, 1).
        with torch.no_grad():
            scale = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / num_levels
        # Quant + dequant (no STE: ZO does not use autograd through this).
        w_q = torch.clamp(torch.round(w / scale), -num_levels - 1, num_levels) * scale
        return F.linear(input, w_q.to(input.dtype), self.bias)


# Modules excluded by the paper (`attr != 'base_model' and attr != 'lm_head'`).
# We also skip nn.Linear that aren't 2-D weight matrices (defensive).
_EXCLUDED_LINEAR_NAMES = ("lm_head",)


def quantize_model(model: nn.Module, wbit: int = 4) -> nn.Module:
    """Recursively replace nn.Linear modules with WeightQuantLinear.

    Skips:
      - lm_head (paper convention; preserve output projection at full precision)
      - WeightQuantLinear instances already quantized (idempotent)
      - 1-D / non-2-D Linears (defensive; HF Llama uses 2-D throughout)
    """
    for name, child in list(model.named_children()):
        if name in _EXCLUDED_LINEAR_NAMES:
            continue
        if isinstance(child, WeightQuantLinear):
            continue
        if isinstance(child, nn.Linear) and child.weight.dim() == 2:
            setattr(model, name, WeightQuantLinear.from_linear(child, wbit=wbit))
        else:
            quantize_model(child, wbit=wbit)
    return model


def count_weight_quant_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, WeightQuantLinear))
