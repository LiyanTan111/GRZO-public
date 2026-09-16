import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math

def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class LoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False, # Not sure if this will affect saving/loading models so just set it to be False
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def set_flipout(self, flip_state, sigma: float):
        self.flip_state = flip_state
        self.sigma = sigma

    def clear_flipout(self):
        self.flip_state = None
        self.sigma = None

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            
            # Check for Flipout state
            if hasattr(self, 'flip_state') and self.flip_state is not None:
                st = self.flip_state
                sgn = float(st.sigma_sign)
                eps = sgn * float(self.sigma)
                
                # Retrieve Flipout noise matrices (ensure correct dtype)
                dtype = x.dtype
                
                # A perturb: ((x * S_A) @ U_A^T) * R_A
                # S_A: (bs, in_features), U_A: (r, in_features), R_A: (bs, r)
                # x: (bs, in_features)
                
                # Handling dimension mismatch (e.g. for classification tasks where input is flattened bs * num_candidates)
                bs_state = st.RA.shape[0]
                bs_input = x.shape[0]
                
                if bs_input != bs_state:
                    if bs_input % bs_state == 0:
                        multiplier = bs_input // bs_state
                        # repeat_interleave ensures (Sample1_Opt1, Sample1_Opt2, Sample2_Opt1, ...) use same noise
                        SA = st.SA.repeat_interleave(multiplier, dim=0).to(dtype)
                        RA = st.RA.repeat_interleave(multiplier, dim=0).to(dtype)
                        SB = st.SB.repeat_interleave(multiplier, dim=0).to(dtype)
                        RB = st.RB.repeat_interleave(multiplier, dim=0).to(dtype)
                    else:
                        raise RuntimeError(f"Flipout Batches mismatch: state bs={bs_state}, input bs={bs_input}")
                else:
                    SA = st.SA.to(dtype)
                    RA = st.RA.to(dtype)
                    SB = st.SB.to(dtype)
                    RB = st.RB.to(dtype)

                if x.dim() == 3:
                    SA = SA.unsqueeze(1)
                    RA = RA.unsqueeze(1)
                    SB = SB.unsqueeze(1)
                    RB = RB.unsqueeze(1)

                UA = st.UA.to(dtype)
                UB = st.UB.to(dtype)
                
                # Note: self.lora_dropout(x) applies dropout.
                # x_d = self.lora_dropout(x)
                # But for Flipout, we apply perturbation on the input to A.
                # x_d has shape (bs, in_features)
                
                x_d = self.lora_dropout(x)
                
                # Standard path h = x_d @ A.T
                h = x_d @ self.lora_A.transpose(0, 1) # (bs, r)
                
                # Noise path
                # (x_d * SA) @ UA.T -> (bs, r)
                # Then * RA -> (bs, r)
                h_noise = (x_d * SA) @ UA.transpose(0, 1)
                h_noise = h_noise * RA
                
                h_perturbed = h + eps * h_noise
                
                # B perturb: ((h * S_B) @ U_B^T) * R_B
                # S_B: (bs, r), U_B: (out_features, r), R_B: (bs, out_features)
                
                # SB, RB already prepared above
                
                # Standard path out = h_perturbed @ B.T
                out = h_perturbed @ self.lora_B.transpose(0, 1) # (bs, out)
                
                # Noise path
                out_noise = (h_perturbed * SB) @ UB.transpose(0, 1)
                out_noise = out_noise * RB
                
                lora_out = out + eps * out_noise
                
                result += lora_out * self.scaling
                
            else:
                if self.r > 0:
                    result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LoRA:

    def __init__(self, model, r, alpha, float16):
        """
        Input:
        r, alpha: LoRA hyperparameters
        float16: Whether the model parameters are float16 or not
        """

        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16

        if model.config.model_type == "opt":
            attention_name = "attn"
        elif model.config.model_type == "roberta":
            attention_name = "attention"
        elif model.config.model_type == "llama":
            attention_name = "self_attn"
        else:
            raise NotImplementedError

        # Insert LoRA
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                logger.info(f"Inject lora to: {key}")
                _, _, attn = find_module(model, key)

                if model.config.model_type == "opt":
                    original_q_weight = attn.q_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data
                    original_v_weight= attn.v_proj.weight.data
                    original_v_bias = attn.v_proj.bias.data
                    attn.q_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=model.config.enable_bias).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=model.config.enable_bias).to(original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight 
                    attn.q_proj.bias.data = original_q_bias
                    attn.v_proj.weight.data = original_v_weight
                    attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "llama":
                    # Llama models don't have bias in attention projections
                    # Llama uses Grouped Query Attention (GQA): k/v projections may have different dims
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    q_out_features = original_q_weight.shape[0]
                    v_out_features = original_v_weight.shape[0]
                    attn.q_proj = LoRALinear(model.config.hidden_size, q_out_features, r=r, lora_alpha=alpha, bias=False).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(model.config.hidden_size, v_out_features, r=r, lora_alpha=alpha, bias=False).to(original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                else:
                    raise NotImplementedError
        
        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False