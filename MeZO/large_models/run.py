import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
# torch >= 2.6 defaults torch.load(weights_only=True), which blocks numpy
# RNG state pickles in HF Trainer checkpoints. Two-pronged fix so resume
# works regardless of which load path HF takes:
#   (1) allowlist the numpy globals that show up in our rng_state pickles
#   (2) monkey-patch torch.load to default weights_only=False (defensive,
#       in case any other consumer hits an unlisted global).
import torch.serialization
import numpy
try:
    torch.serialization.add_safe_globals([
        numpy._core.multiarray._reconstruct,
        numpy.ndarray,
        numpy.dtype,
    ])
except Exception:
    pass
_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
from trainer import OurTrainer
import random

@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 

    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO
    zo_optimizer: str = "mezo" # mezo | mezo_lozo | mezo_sparse | mezo_quzo | fzoo | flipout (GRZO) | grzo_lozo | grzo_lozo_strict | grzo_sparse | grzo_quzo
    fzoo_n: int = 8 # FZOO: number of perturbations per step
    zo_group_size: int = 1 # MeZO-family: number of z samples averaged per step (paper-canonical = 1)
    u_distribution: str = "gaussian" # base noise distribution for Flipout: gaussian or rademacher
    estimation_side: str = "two" # estimation type for Flipout: one (single-sided) or two (two-sided)
    # GRZO + SparseMeZO (zo_optimizer="grzo_sparse") parameters
    sparse_ratio: float = 0.25  # fraction of each Linear weight kept active (perturbed + updated); rest is masked out
    sparse_rule: str = "small"   # which weights stay active: "small" (|W| <= thr, SparseMeZO rule) or "large" (|W| > thr)

    # GRZO + LOZO (zo_optimizer="grzo_lozo") parameters
    lozo_rank: int = 8                  # rank r of the (u@v) low-rank Gaussian noise that replaces full-rank U in flipout
    lozo_step_interval: int = 50        # LOZO default — resample v every N steps; u is resampled every step

    # GRZO + QuZO (zo_optimizer="grzo_quzo") parameters
    quant_bits: int = 4                 # symmetric quant bit-width for the base noise U (2/4/8 typical)
    # QuZO Algorithm 1 weight quantization (paper qft mode). When > 0, wraps
    # every transformer-block nn.Linear with fake-quant on forward at this bit
    # width. Underlying weight stays fp16/fp32 (perturbed by ZO), but compute
    # uses int-Wbit. Set to 4 to match paper WBIT=4. Set to 0 to disable.
    quzo_weight_bits: int = 0

    # GRZO normalization safeguards (apply to all flipout-based GRZO variants).
    # Both default to 0 = off (preserves vanilla GRZO behavior). Useful for
    # low-rank / sparse / quantized combos where weak per-batch signal can
    # cause group-relative normalization to amplify noise.
    adv_std_floor: float = 0.0          # clamp std(delta) to at least this before normalizing; e.g. 1e-3
    adv_clip: float = 0.0               # clip per-example advantages to [-c, +c]; e.g. 2.0 or 3.0

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag
    overwrite_output_dir: bool = False # overwrite output directory
    past_index: int = -1 # past index for keeping track of past state

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                # Head tuning
                from ht_opt import OPTForCausalLM
                model = OPTForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16

                # Under torchrun / DDP each rank only owns one GPU. If LOCAL_RANK
                # is set, pin the whole model to that one GPU instead of letting
                # `device_map='auto'` spread it across all visible GPUs (which
                # would cause NCCL "Duplicate GPU detected" errors).
                # For single-GPU runs we also pin to cuda:0 so accelerate doesn't
                # offload anything to CPU/disk (which then breaks HF Trainer's
                # later `model.to(device)` call: "You can't move a model that has
                # some modules offloaded").
                _local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
                if _local_rank >= 0:
                    device_map = {"": _local_rank}
                    max_memory_arg = None
                elif torch.cuda.device_count() == 1:
                    device_map = {"": 0}
                    max_memory_arg = None
                else:
                    device_map = "auto"
                    max_memory_arg = {i: f'{free_in_GB-5}GB' for i in range(torch.cuda.device_count())}

                kwargs = {
                    "device_map": device_map,
                    "torch_dtype": torch_dtype,
                    "offload_folder": os.path.join(self.args.output_dir, "offload_weights"),
                }
                if max_memory_arg is not None:
                    kwargs["max_memory"] = max_memory_arg
                if self.args.load_int8:
                    kwargs["load_in_8bit"] = True

                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    **kwargs
                )
            model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0
        
        if "llama" in self.args.model_name.lower():
            # LLaMA padding token
            tokenizer.pad_token_id = 0 # technically <unk>

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        # Freeze lm_head to avoid OOM during flipout: lm_head has shape (vocab_size=128256, hidden)
        # and its flipout hook creates multiple (B, S, vocab_size) tensors simultaneously (~7 GB each).
        # Freezing excludes it from flipout module selection (trainer.py filters by requires_grad).
        if hasattr(model, 'lm_head'):
            for p in model.lm_head.parameters():
                p.requires_grad = False
            logger.info("Froze lm_head parameters to reduce flipout memory usage.")

        # QuZO Algorithm 1 weight quantization. Wraps every transformer-block
        # nn.Linear in WeightQuantLinear (forward fake-quant at Wbit precision);
        # lm_head stays in full precision. Underlying weight is still fp16 so
        # ZO perturbations + hooks behave identically to the non-quant case.
        if getattr(self.args, "quzo_weight_bits", 0) > 0:
            import sys as _sys, os as _os
            _repo = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
            if _repo not in _sys.path:
                _sys.path.insert(0, _repo)
            from optimizers.quzo_wquant import quantize_model, count_weight_quant_linears
            wbit = int(self.args.quzo_weight_bits)
            quantize_model(model, wbit=wbit)
            logger.info(f"QuZO weight-quant: wrapped {count_weight_quant_linears(model)} nn.Linear modules at W{wbit}.")

        return model, tokenizer


    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    num_opts = len(sample.candidates) if sample.candidates is not None else 1
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": num_opts} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        trainer = OurTrainer(
            model=self.model, 
            args=self.args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=last_checkpoint) 

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    args = parse_args()

    # If launched via torchrun, bind this process to its local GPU so subsequent
    # CUDA ops (and `device_map='auto'`) don't see/grab other ranks' GPUs.
    _local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if _local_rank >= 0:
        torch.cuda.set_device(_local_rank)

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)

if __name__ == "__main__": 
    main()
