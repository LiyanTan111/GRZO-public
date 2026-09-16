import json
import os
import contextlib
from typing import Optional, Union
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
import logging
import time
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
import transformers
from typing import Optional, Union, List, Dict, Any
import signal
from subprocess import call
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
InputDataClass = NewType("InputDataClass", Any)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase



logger = logging.getLogger(__name__)


def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, return_per_example_loss=False, **kwargs):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens 
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    - return_per_example_loss: if True, return loss per example (tensor of size bsz) instead of scalar mean.
    """
    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs
    logits = outputs.logits

    loss = None
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None:
        # Memory-efficient classification: avoid materializing full (B, S-1, vocab) tensors
        # (which can be ~10 GiB for large batches with big vocabularies).
        # Instead, for each row extract only the option-position logits (tiny), compute
        # log_softmax there, then gather the label log-probs.
        shift_logits_view = logits[..., :-1, :]  # non-contiguous view — no copy
        mask = shift_labels != -100  # (B_flat, S-1)

        selected_log_probs = torch.zeros(
            shift_labels.shape[0], device=shift_labels.device, dtype=torch.float32
        )
        for b in range(shift_labels.shape[0]):
            opt_pos = mask[b]        # (S-1,) bool
            n_opt = opt_pos.sum()
            if n_opt == 0:
                continue
            # shift_logits_view[b] is contiguous (stride (vocab,1)); boolean-mask to get option rows
            lgt_b = shift_logits_view[b][opt_pos].float()   # (n_opt, vocab)
            lbl_b = shift_labels[b][opt_pos].clamp(min=0)   # (n_opt,) — never -100 here
            lp_b = F.log_softmax(lgt_b, dim=-1)              # (n_opt, vocab)
            del lgt_b
            selected_log_probs[b] = (
                lp_b.gather(-1, lbl_b.unsqueeze(-1)).squeeze(-1).sum() / n_opt.float()
            )
        del shift_logits_view  # allow logits memory to be freed after this point

        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            # If returning per example data, we need a list or tensor? 
            # Variable size options make tensor difficult without padding?
            # But the user's batch usually has uniform options for standard tasks (like RTE, SST2).
            # For complex tasks, we might need to handle it.
            # Assuming uniform here for Flipout for now or simple accumulation.
            
            loss_list = []
            start_id = 0
            for i in range(len(num_options)):
                end_id = start_id + num_options[i]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0) # (1, num_options)
                
                # Get labels for this example
                _labels_slice = labels[start_id:end_id]
                if len(_labels_slice) == 0:
                    # Skip empty slices (shouldn't happen but safeguard)
                    start_id = end_id
                    continue
                _labels = _labels_slice[0].unsqueeze(0) # (1)
                
                # Check reduction
                l = loss_fct(_logits, _labels) 
                loss_list.append(l)
                start_id = end_id
            
            if len(loss_list) == 0:
                # Fallback: if no valid losses, return zero loss
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            elif return_per_example_loss:
                loss = torch.stack(loss_list) # (bsz,)
            else:
                loss = torch.stack(loss_list).mean()
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(-1, num_options) # (bsz, num_options)
            labels = labels.view(-1, num_options)[:, 0] # Labels repeat so we only take the first one
            
            if return_per_example_loss:
                loss_fct_none = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct_none(selected_log_probs, labels) # (bsz,)
            else:
                loss = loss_fct(selected_log_probs, labels)
    else:
        # Standard LM training (not classification) — generative tasks (SQuAD, DROP, etc.)
        shift_logits = logits[..., :-1, :].contiguous()
        if return_per_example_loss:
            loss_fct_none = CrossEntropyLoss(ignore_index=-100, reduction='none')
            # (bsz * seq_len)
            l = loss_fct_none(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            l = l.view(shift_logits.size(0), shift_logits.size(1)) # (bsz, seq_len)
            # Sum over sequence
            mask = (shift_labels != -100).float()
            l = (l * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
            loss = l
        else:
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False, generation=False, generation_with_gold=False, max_new_tokens=None):
    """
    Encode prompts for eval_sample
    Input: 
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, max_length: tokenizer and max length
    - sfc: generate prompts for calibration (surface form competition; https://arxiv.org/abs/2104.08315)
    - icl_sfc: generate prompts for ICL version calibration
    - generation: whether it is an generation task
    - generation_with_gold: whether to include the generation-task gold answers (for training)
    - max_new_tokens: max number of new tokens to generate so that we can save enough space 
      (only for generation tasks)
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """

    # Demonstrations for ICL
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    train_prompts = task.train_sep.join(train_prompts).strip()
    
    # sfc or icl_sfc indicates that this example is used for calibration
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc; verbalize_fn = template.verbalize_sfc
    else: 
        encode_fn = template.encode; verbalize_fn = template.verbalize 
            
    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    if not generation:
        # We generate one prompt for each candidate (different classes in classification)
        # or different choices in multiple-choice tasks
        verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]

        if sfc:
            # Without demonstrations
            final_prompts = verbalized_eval_prompts 
        else:
            # With demonstrations
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
        else:
            option_lens = [0]
            final_prompts = [(train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')]

    # Tokenize 
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    # Truncate (left truncate as demonstrations are less important)
    if generation and max_new_tokens is not None:
        max_length = max_length - max_new_tokens

    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    # Detect whether the tokenizer prepends a BOS token. Older slow tokenizers
    # (e.g. LlamaTokenizer) expose `add_bos_token` directly; fast tokenizers
    # (PreTrainedTokenizerFast, used by Llama-3) do not, so infer from output.
    if hasattr(tokenizer, "add_bos_token"):
        adds_bos = tokenizer.add_bos_token
    else:
        bos_id = getattr(tokenizer, "bos_token_id", None)
        adds_bos = bos_id is not None and len(encodings[0]) > 0 and encodings[0][0] == bos_id
    if adds_bos:
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]
   
    return encodings, option_lens
 


@dataclass
class ICLCollator:
    """
    Collator for ICL
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        
        pad_id = self.tokenizer.pad_token_id

        pad_ids = {"input_ids": pad_id, "attention_mask": 0, "sfc_input_ids": pad_id, "sfc_attention_mask": 0, "labels": pad_id}
        for key in first:
            pp = pad_ids[key]
            lens = [len(f[key]) for f in features]
            max_len = max(lens)
            feature = np.stack([np.pad(f[key], (0, max_len - lens[i]), "constant", constant_values=(0, pp)) for i, f in enumerate(features)])
            padded_feature = torch.from_numpy(feature).long()
            batch[key] = padded_feature
            
        return batch


@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Preserve num_options before flattening
        num_options_list = [f[0]["num_options"] for f in features if len(f) > 0 and "num_options" in f[0]]
        option_len_list = [[ff["option_len"] for ff in f if "option_len" in ff] for f in features]
        
        features = [ff for f in features for ff in f]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        # Add num_options back to batch. Wrap as torch tensors so they survive
        # BatchEncoding.to(device) (which drops non-Tensor entries).
        import torch
        if len(num_options_list) > 0:
            batch["num_options"] = torch.tensor(num_options_list, dtype=torch.long)
        if len(option_len_list) > 0 and len(option_len_list[0]) > 0:
            batch["option_len"] = torch.tensor(
                [ol for sublist in option_len_list for ol in sublist], dtype=torch.long
            )
        return batch


@dataclass
class NondiffCollator(DataCollatorMixin):
    """
    Collator for non-differentiable objectives
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k != "gold"} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        if "gold" in features[0]:
            batch["gold"] = [feature["gold"] for feature in features]
        
        return batch
        

class SIGUSR1Callback(transformers.TrainerCallback):
    """
    This callback is used to save the model when a SIGUSR1 signal is received
    (SLURM stop signal or a keyboard interruption signal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def write_predictions_to_file(final_preds, output):
    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, cls=EnhancedJSONEncoder) + "\n")


def write_metrics_to_file(metrics, output):
    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
    json.dump(metrics, open(output, "w"), cls=EnhancedJSONEncoder, indent=4)