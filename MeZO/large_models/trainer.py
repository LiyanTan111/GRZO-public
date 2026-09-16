# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import os
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "optimizers")))

# --- Production-time profiling instrument (no-op unless GRZO_PROFILE_OUT is set) ---
import time as _time, json as _json
_PROFILE_OUT = os.environ.get("GRZO_PROFILE_OUT")
_PROFILE_WARMUP = int(os.environ.get("GRZO_PROFILE_WARMUP", "5"))
_PROFILE_DATA = []
_PROFILE_STEP = 0

class _pf:
    """Context manager: torch.cuda.synchronize() + perf_counter + peak_memory tracking, append to _PROFILE_DATA."""
    def __init__(self, name):
        self.name = name
        self.active = (_PROFILE_OUT is not None) and (_PROFILE_STEP >= _PROFILE_WARMUP)
    def __enter__(self):
        if self.active:
            import torch as _t
            _t.cuda.synchronize()
            _t.cuda.reset_peak_memory_stats()
            self.mem_start = _t.cuda.memory_allocated()
            self.t0 = _time.perf_counter()
        return self
    def __exit__(self, *a):
        if self.active:
            import torch as _t
            _t.cuda.synchronize()
            elapsed_ms = (_time.perf_counter() - self.t0) * 1000.0
            peak_mb = _t.cuda.max_memory_allocated() / 1024**2
            delta_mb = (_t.cuda.memory_allocated() - self.mem_start) / 1024**2
            _PROFILE_DATA.append({
                "name": self.name, "step": _PROFILE_STEP,
                "t_ms": elapsed_ms,
                "peak_MB": peak_mb,
                "delta_MB": delta_mb,
            })

def _pf_flush():
    """Called at end-of-training to write JSON."""
    if _PROFILE_OUT is None or not _PROFILE_DATA:
        return
    try:
        import torch as _t
        rank = _t.distributed.get_rank() if _t.distributed.is_initialized() else 0
    except Exception:
        rank = 0
    path = _PROFILE_OUT.replace(".json", f"_rank{rank}.json")
    with open(path, "w") as f:
        _json.dump(_PROFILE_DATA, f)
    print(f"[grzo-profile] rank{rank} wrote {len(_PROFILE_DATA)} records to {path}", file=sys.stderr)

import atexit as _atexit
_atexit.register(_pf_flush)
# --- end profile instrument ---
try:
    from flipout import grzo_flipout_step, grzo_flipout_step_ddp
except ImportError:
    print("Warning: Could not import grzo_flipout_step / grzo_flipout_step_ddp")
    grzo_flipout_step = None
    grzo_flipout_step_ddp = None

# Wrap grzo step fns with profile timer if profile mode enabled
def _wrap_grzo_for_profile(fn):
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _PROFILE_STEP
        with _pf("grzo_step_total"):
            result = fn(*args, **kwargs)
        if _PROFILE_OUT is not None:
            _PROFILE_STEP += 1
            if len(_PROFILE_DATA) % 100 == 0 and len(_PROFILE_DATA) > 0:
                _pf_flush()
        return result
    return wrapper

if _PROFILE_OUT is not None:
    if grzo_flipout_step_ddp is not None:
        grzo_flipout_step_ddp = _wrap_grzo_for_profile(grzo_flipout_step_ddp)
    if grzo_flipout_step is not None:
        grzo_flipout_step = _wrap_grzo_for_profile(grzo_flipout_step)

try:
    from lora import LoRALinear
except ImportError:
    print("Warning: Could not import LoRALinear")
    LoRALinear = None

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
# from transformers.integrations import (  # isort: split
#     default_hp_search_backend,
#     get_reporting_integration_callbacks,
#     hp_params,
#     is_fairscale_available,
#     is_optuna_available,
#     is_ray_tune_available,
#     is_sigopt_available,
#     is_wandb_available,
#     run_hp_search_optuna,
#     run_hp_search_ray,
#     run_hp_search_sigopt,
#     run_hp_search_wandb,
# )
def is_fairscale_available():
    return False

def hp_params(assignments):
    return assignments

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
is_torch_greater_or_equal_than_1_10 = True
is_torch_less_than_1_11 = False
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

class ShardedDDPOption:
    SIMPLE = "simple"
    ZERO_DP_2 = "zero_dp_2"
    ZERO_DP_3 = "zero_dp_3"
    OFFLOAD = "offload"

def default_hp_space(*args, **kwargs):
    return {}

from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    # is_torch_tpu_available,
    logging,
)

# Compatibility shims for removed functions in newer transformers versions
def is_ipex_available():
    return False

def is_torch_tensorrt_fx_available():
    return False

def is_torch_tpu_available(check_device=False):
    return False

from transformers.utils.generic import ContextManagers


_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                        
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4 # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial", random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1: # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            getattr(self, "sharded_ddp", None) is not None
            and getattr(self, "sharded_ddp", None) != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or getattr(self, "fsdp", None) is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.stateful_callbacks["TrainerControl"] = self.control.state()
        self.state.is_hyper_param_search = trial is not None
        # Propagate logging / eval / save step intervals from args to state — HF's
        # DefaultFlowCallback reads them off `state` (not `args`) and otherwise
        # defaults all three to 500, which silently throttles intra-training logs.
        self.state.logging_steps = args.logging_steps
        self.state.eval_steps = args.eval_steps
        self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # MeZO added: Initialize Flipout gradient accumulation storage
        if args.trainer == "zo" and getattr(args, "zo_optimizer", "mezo") == "flipout" and args.gradient_accumulation_steps > 1:
            self.flipout_grad_storage = {}
            logger.info(f"Initialized Flipout gradient accumulation with {args.gradient_accumulation_steps} steps")
        else:
            self.flipout_grad_storage = None

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # MeZO added: estimate gradient
                if args.trainer == "zo":
                    if getattr(args, "zo_optimizer", "mezo") == "flipout":
                         # Identify LoRA modules or Full FT modules
                        lora_modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
                        if not lora_modules:
                            # If no LoRA modules, collect all trainable modules for full fine-tuning:
                            # Linear, LayerNorm, RMSNorm (for Llama), and Embedding layers
                            lora_modules = [
                                m for m in model.modules() 
                                if (isinstance(m, (nn.Linear, nn.LayerNorm, nn.Embedding)) or 
                                    type(m).__name__ in ['LlamaRMSNorm', 'RMSNorm']) and 
                                   any(p.requires_grad for p in m.parameters())
                            ]
                            if step == 0:  # Only log once
                                logger.info(f"Full FT mode: Found {len(lora_modules)} trainable modules for Flipout")
                        
                        # Add attention modules that hold prefix_keys/prefix_values parameters
                        prefix_mods = [
                            m for m in model.modules()
                            if (hasattr(m, 'prefix_keys') and m.prefix_keys.requires_grad) or
                               (hasattr(m, 'prefix_values') and m.prefix_values.requires_grad)
                        ]
                        if prefix_mods:
                            lora_modules = lora_modules + prefix_mods
                            if step == 0:
                                logger.info(f"Prefix tuning: found {len(prefix_mods)} attention modules with trainable prefix params")

                        if not lora_modules:
                            logger.warning("No trainable modules found for Flipout optimization!")

                        # Determine accumulation step (1-indexed)
                        accum_step = (step % args.gradient_accumulation_steps) + 1
                        is_accumulating = (args.gradient_accumulation_steps > 1)
                        
                        # Run Flipout step with gradient accumulation support.
                        # Under multi-GPU DDP we use the DDP-correct path; otherwise
                        # the single-GPU implementation.
                        import torch.distributed as _dist
                        _use_ddp = (
                            _dist is not None
                            and _dist.is_available()
                            and _dist.is_initialized()
                            and _dist.get_world_size() > 1
                            and grzo_flipout_step_ddp is not None
                        )
                        _flipout_fn = grzo_flipout_step_ddp if _use_ddp else grzo_flipout_step
                        tr_loss_step_val = _flipout_fn(
                            model=model,
                            inputs=inputs,
                            lora_modules=lora_modules,
                            sigma=args.zo_eps,
                            lr=self._get_learning_rate(),
                            eps=1e-8,
                            u_distribution=args.u_distribution,
                            estimation_side=args.estimation_side,
                            accumulate_grad=is_accumulating,
                            grad_accumulation_storage=self.flipout_grad_storage if is_accumulating else None,
                            accum_step=accum_step,
                            total_accum_steps=args.gradient_accumulation_steps,
                            adv_std_floor=getattr(args, "adv_std_floor", 0.0),
                            adv_clip=getattr(args, "adv_clip", 0.0),
                        )
                        # Ensure tr_loss_step is a tensor for compatibility with Trainer's expectations below
                        tr_loss_step = torch.tensor(tr_loss_step_val, device=model.device)

                    elif getattr(args, "zo_optimizer", "mezo") == "grzo_quzo":
                        # GRZO + QuZO: per-tensor symmetric quant of the base
                        # noise U to args.quant_bits (default 4). Same flipout
                        # estimator otherwise.
                        lora_modules = [
                            m for m in model.modules()
                            if (isinstance(m, (nn.Linear, nn.LayerNorm, nn.Embedding)) or
                                type(m).__name__ in ['LlamaRMSNorm', 'RMSNorm']) and
                               any(p.requires_grad for p in m.parameters())
                        ]
                        q_bits = int(getattr(args, "quant_bits", 4))
                        if step == 0:
                            logger.info(f"GRZO+QuZO: quant_bits={q_bits} on {len(lora_modules)} modules")
                        accum_step = (step % args.gradient_accumulation_steps) + 1
                        is_accumulating = (args.gradient_accumulation_steps > 1)
                        import torch.distributed as _dist
                        _use_ddp = (
                            _dist is not None and _dist.is_available()
                            and _dist.is_initialized() and _dist.get_world_size() > 1
                            and grzo_flipout_step_ddp is not None
                        )
                        _fn = grzo_flipout_step_ddp if _use_ddp else grzo_flipout_step
                        tr_loss_step_val = _fn(
                            model=model,
                            inputs=inputs,
                            lora_modules=lora_modules,
                            sigma=args.zo_eps,
                            lr=self._get_learning_rate(),
                            eps=1e-8,
                            u_distribution=args.u_distribution,
                            estimation_side=args.estimation_side,
                            accumulate_grad=is_accumulating,
                            grad_accumulation_storage=self.flipout_grad_storage if is_accumulating else None,
                            accum_step=accum_step,
                            total_accum_steps=args.gradient_accumulation_steps,
                            quant_bits=q_bits,
                            adv_std_floor=getattr(args, "adv_std_floor", 0.0),
                            adv_clip=getattr(args, "adv_clip", 0.0),
                        )
                        tr_loss_step = torch.tensor(tr_loss_step_val, device=model.device)

                    elif getattr(args, "zo_optimizer", "mezo") in ("grzo_lozo", "grzo_lozo_strict"):
                        # GRZO + LOZO: replace U in flipout with rank-r `u @ v`.
                        # u resampled every step, v every `lozo_step_interval` steps.
                        # Variant `grzo_lozo_strict`: also drop per-example column
                        # sign s_i so the aggregate update stays in span(V).
                        lora_modules = [
                            m for m in model.modules()
                            if (isinstance(m, (nn.Linear, nn.LayerNorm, nn.Embedding)) or
                                type(m).__name__ in ['LlamaRMSNorm', 'RMSNorm']) and
                               any(p.requires_grad for p in m.parameters())
                        ]
                        r_max = int(getattr(args, "lozo_rank", 8))
                        v_interval = max(1, int(getattr(args, "lozo_step_interval", 50)))
                        # Initialize / refresh v at step_interval boundaries; u every step.
                        if not hasattr(self, "_grzo_lozo_v_cache"):
                            self._grzo_lozo_v_cache = {}
                        refresh_v = (self.state.global_step % v_interval == 0)
                        # Deterministic per-step seeds so (u,v) are identical across DDP ranks.
                        _gs = int(self.state.global_step)
                        _v_seed = 0xC0FFEE + (_gs // v_interval)
                        _u_seed = 0xBEEF + _gs
                        lozo_uv = {}
                        idx = 0
                        for _m in lora_modules:
                            if not isinstance(_m, nn.Linear):
                                continue
                            dout, din = _m.weight.shape
                            r = min(r_max, dout, din)
                            if refresh_v or _m not in self._grzo_lozo_v_cache:
                                g_v = torch.Generator(device=_m.weight.device)
                                g_v.manual_seed((_v_seed * 1000003 + idx) % (2**31))
                                self._grzo_lozo_v_cache[_m] = torch.randn(
                                    r, din, generator=g_v,
                                    device=_m.weight.device, dtype=torch.float32,
                                )
                            g_u = torch.Generator(device=_m.weight.device)
                            g_u.manual_seed((_u_seed * 1000003 + idx) % (2**31))
                            _u = torch.randn(dout, r, generator=g_u, device=_m.weight.device, dtype=torch.float32)
                            lozo_uv[_m] = (_u, self._grzo_lozo_v_cache[_m])
                            idx += 1
                        if step == 0:
                            logger.info(
                                f"GRZO+LOZO: r={r_max}, v_interval={v_interval}, "
                                f"{len(lozo_uv)} Linear modules"
                            )
                        accum_step = (step % args.gradient_accumulation_steps) + 1
                        is_accumulating = (args.gradient_accumulation_steps > 1)
                        import torch.distributed as _dist
                        _use_ddp = (
                            _dist is not None and _dist.is_available()
                            and _dist.is_initialized() and _dist.get_world_size() > 1
                            and grzo_flipout_step_ddp is not None
                        )
                        _fn = grzo_flipout_step_ddp if _use_ddp else grzo_flipout_step
                        tr_loss_step_val = _fn(
                            model=model,
                            inputs=inputs,
                            lora_modules=lora_modules,
                            sigma=args.zo_eps,
                            lr=self._get_learning_rate(),
                            eps=1e-8,
                            u_distribution=args.u_distribution,
                            estimation_side=args.estimation_side,
                            accumulate_grad=is_accumulating,
                            grad_accumulation_storage=self.flipout_grad_storage if is_accumulating else None,
                            accum_step=accum_step,
                            total_accum_steps=args.gradient_accumulation_steps,
                            lozo_uv=lozo_uv,
                            lozo_strict=(getattr(args, "zo_optimizer", "mezo") == "grzo_lozo_strict"),
                            adv_std_floor=getattr(args, "adv_std_floor", 0.0),
                            adv_clip=getattr(args, "adv_clip", 0.0),
                        )
                        tr_loss_step = torch.tensor(tr_loss_step_val, device=model.device)

                    elif getattr(args, "zo_optimizer", "mezo") == "grzo_sparse":
                        # GRZO + SparseMeZO: per-example flipout perturbation
                        # restricted to the top-(sparse_ratio) fraction of each
                        # Linear weight matrix by current magnitude.
                        lora_modules = [
                            m for m in model.modules()
                            if (isinstance(m, (nn.Linear, nn.LayerNorm, nn.Embedding)) or
                                type(m).__name__ in ['LlamaRMSNorm', 'RMSNorm']) and
                               any(p.requires_grad for p in m.parameters())
                        ]
                        # `sparse_ratio` is the fraction of Linear weights that stay
                        # active; `sparse_rule` picks which ones:
                        #   "small" (default, SparseMeZO rule): mask = |W| <= thr
                        #   "large":                           mask = |W| >  thr
                        _rule = str(getattr(args, "sparse_rule", "small")).strip().lower()
                        if _rule not in ("large", "small"):
                            _rule = "small"
                        ratio = float(getattr(args, "sparse_ratio", 0.25))
                        ratio = max(0.0, min(1.0, ratio))
                        if not hasattr(self, "_grzo_sparse_thresholds"):
                            self._grzo_sparse_thresholds = {}
                            for _m in lora_modules:
                                if isinstance(_m, nn.Linear):
                                    _n = _m.weight.numel()
                                    if _rule == "large":
                                        _k = max(1, int((1.0 - ratio) * _n))
                                    else:
                                        _k = max(1, int(ratio * _n))
                                    if _k >= _n:
                                        continue
                                    _thr = torch.kthvalue(_m.weight.detach().abs().view(-1), _k)[0]
                                    self._grzo_sparse_thresholds[_m] = _thr
                            self._grzo_sparse_rule = _rule
                            logger.info(
                                f"GRZO+Sparse: cached thresholds (rule={_rule}-weight) for {len(self._grzo_sparse_thresholds)} Linear modules"
                            )
                        sparse_masks = {}
                        _rule_cached = getattr(self, "_grzo_sparse_rule", _rule)
                        for _m, _thr in self._grzo_sparse_thresholds.items():
                            if _rule_cached == "large":
                                sparse_masks[_m] = (_m.weight.detach().abs() > _thr)
                            else:
                                sparse_masks[_m] = (_m.weight.detach().abs() <= _thr)
                        if step == 0:
                            _active = sum(int(m_.sum().item()) for m_ in sparse_masks.values())
                            _total = sum(m_.numel() for m_ in sparse_masks.values())
                            logger.info(
                                f"GRZO+Sparse: rule={_rule_cached}, ratio={ratio}, {_active}/{_total} active "
                                f"({100*_active/max(_total,1):.1f}%) across {len(sparse_masks)} Linear modules"
                            )
                        accum_step = (step % args.gradient_accumulation_steps) + 1
                        is_accumulating = (args.gradient_accumulation_steps > 1)
                        import torch.distributed as _dist
                        _use_ddp = (
                            _dist is not None and _dist.is_available()
                            and _dist.is_initialized() and _dist.get_world_size() > 1
                            and grzo_flipout_step_ddp is not None
                        )
                        _fn = grzo_flipout_step_ddp if _use_ddp else grzo_flipout_step
                        tr_loss_step_val = _fn(
                            model=model,
                            inputs=inputs,
                            lora_modules=lora_modules,
                            sigma=args.zo_eps,
                            lr=self._get_learning_rate(),
                            eps=1e-8,
                            u_distribution=args.u_distribution,
                            estimation_side=args.estimation_side,
                            accumulate_grad=is_accumulating,
                            grad_accumulation_storage=self.flipout_grad_storage if is_accumulating else None,
                            accum_step=accum_step,
                            total_accum_steps=args.gradient_accumulation_steps,
                            sparse_masks=sparse_masks,
                            adv_std_floor=getattr(args, "adv_std_floor", 0.0),
                            adv_clip=getattr(args, "adv_clip", 0.0),
                        )
                        tr_loss_step = torch.tensor(tr_loss_step_val, device=model.device)

                    else:
                        tr_loss_step = self.zo_step(model, inputs)
                else:
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer == "zo":
                        if getattr(args, "zo_optimizer", "mezo") in ("flipout", "grzo_sparse", "grzo_lozo", "grzo_lozo_strict", "grzo_quzo"):
                            # GRZO-family updates are applied inside their step functions
                            self.lr_scheduler.step()
                        else:
                            self.zo_update(model)
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    if self.state.global_step == 10:
                        _peak_mib = torch.cuda.max_memory_allocated() / 1024**2
                        logger.info(f"[PEAK_MEMORY] step=10 peak_allocated={_peak_mib:.0f} MiB")
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                # [Fix for single-GPU usage] Only barrier if distributed is initialized
                if dist.is_initialized():
                    dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    ############## MeZO ##############


    def _baseline_modify_z(self, name, param, z):
        """Apply baseline-optimizer-specific transformation to a freshly sampled
        Gaussian z. Used by `mezo_lozo`, `mezo_sparse`, `mezo_quzo` — the
        pure-MeZO baselines that compare against our GRZO+X combinations.

        Caches (sparse masks, low-rank u/v) are set up in `_init_baseline_state`
        at the start of each `zo_step` so they're identical across the +1, -2,
        +1 perturb calls and the matching zo_update call.
        """
        opt = getattr(self.args, "zo_optimizer", "mezo")
        if opt == "mezo_sparse":
            mask = getattr(self, "_mezo_sparse_masks", {}).get(name)
            if mask is not None:
                return z * mask.to(device=z.device, dtype=z.dtype)
            return z
        if opt == "mezo_quzo":
            from flipout import _zo_quant_dequant_sym  # noqa: WPS433
            return _zo_quant_dequant_sym(z, int(getattr(self.args, "quant_bits", 4)))
        if opt == "mezo_lozo":
            # Only Linear 2D weights get rank-r factor; bias/norm stay Gaussian.
            if param.dim() == 2:
                uv = getattr(self, "_mezo_lozo_uv", {}).get(name)
                if uv is not None:
                    u, v = uv
                    r = u.shape[1]
                    return ((u.to(z.dtype) @ v.to(z.dtype)) / (r ** 0.5))
            return z
        return z

    def _init_baseline_state(self):
        """Per-step setup of the per-Linear caches used by the mezo_* baselines.
        Called once at the top of every `zo_step`."""
        opt = getattr(self.args, "zo_optimizer", "mezo")
        if opt == "mezo_sparse":
            # `sparse_ratio` = fraction of weights ACTIVE (smallest ones per
            # SparseMeZO paper). Default 0.20 = paper's sparsity 0.80.
            ratio = float(getattr(self.args, "sparse_ratio", 0.20))
            ratio = max(0.0, min(1.0, ratio))
            if not hasattr(self, "_mezo_sparse_thresholds"):
                self._mezo_sparse_thresholds = {}
                for name, p in self.named_parameters_to_optim:
                    if p.dim() == 2 and "norm" not in name.lower() and "embed" not in name.lower():
                        n = p.numel()
                        k_keep = max(1, int(ratio * n))
                        if k_keep < n:
                            # k-th smallest |W| → threshold for small weights
                            self._mezo_sparse_thresholds[name] = torch.kthvalue(
                                p.detach().abs().view(-1), k_keep
                            )[0]
                logger.info(
                    f"mezo_sparse: cached thresholds (small-weight rule) for {len(self._mezo_sparse_thresholds)} Linear params"
                )
            # Mask = small weights (|W| <= threshold).
            self._mezo_sparse_masks = {
                name: (p.detach().abs() <= self._mezo_sparse_thresholds[name])
                for name, p in self.named_parameters_to_optim
                if name in self._mezo_sparse_thresholds
            }
        elif opt == "mezo_lozo":
            r_max = int(getattr(self.args, "lozo_rank", 8))
            v_interval = max(1, int(getattr(self.args, "lozo_step_interval", 50)))
            if not hasattr(self, "_mezo_lozo_v_cache"):
                self._mezo_lozo_v_cache = {}
            refresh_v = (self.state.global_step % v_interval == 0)
            self._mezo_lozo_uv = {}
            # Use deterministic seeds based on global_step so u,v are identical
            # across DDP ranks (otherwise ranks diverge with different noise).
            gs = int(self.state.global_step)
            v_seed_base = 0xC0FFEE + (gs // v_interval)
            u_seed_base = 0xBEEF + gs
            idx = 0
            for name, p in self.named_parameters_to_optim:
                if p.dim() != 2:
                    continue
                if "norm" in name.lower() or "embed" in name.lower():
                    continue
                dout, din = p.shape
                r = min(r_max, dout, din)
                if refresh_v or name not in self._mezo_lozo_v_cache:
                    g_v = torch.Generator(device=p.device)
                    g_v.manual_seed((v_seed_base * 1000003 + idx) % (2**31))
                    self._mezo_lozo_v_cache[name] = torch.randn(
                        r, din, generator=g_v, device=p.device, dtype=torch.float32,
                    )
                g_u = torch.Generator(device=p.device)
                g_u.manual_seed((u_seed_base * 1000003 + idx) % (2**31))
                u = torch.randn(dout, r, generator=g_u, device=p.device, dtype=torch.float32)
                self._mezo_lozo_uv[name] = (u, self._mezo_lozo_v_cache[name])
                idx += 1

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z = self._baseline_modify_z(name, param, z)
            param.data = param.data + scaling_factor * z * self.args.zo_eps


    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # Under DDP each rank's `loss` is computed on its local minibatch
            # slice. For scalar-gradient MeZO (and its baselines) we need the
            # GLOBAL mean loss so projected_grad is identical on all ranks and
            # the subsequent update keeps weights in sync.
            import torch.distributed as _dist
            if (
                _dist is not None
                and _dist.is_available()
                and _dist.is_initialized()
                and _dist.get_world_size() > 1
            ):
                if not torch.is_tensor(loss):
                    loss = torch.tensor(loss, device=next(model.parameters()).device)
                _dist.all_reduce(loss, op=_dist.ReduceOp.AVG)
        return loss.detach()


    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.processing_class.encode(args.eos_token, add_special_tokens=False)[-1], self.processing_class.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.processing_class.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)


    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO with multiple sampling. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Get group_size (number of samples for gradient estimation)
        group_size = getattr(args, "zo_group_size", 1)
        zo_opt = getattr(args, "zo_optimizer", "mezo")

        # Per-step setup for mezo_lozo / mezo_sparse / mezo_quzo baselines.
        self._init_baseline_state()

        # Store random seeds for each sample
        self.zo_random_seeds = [np.random.randint(1000000000) for _ in range(group_size)]

        if zo_opt == "fzoo":
            # FZOO (Yan et al. ICLR 2026): N forwards with Rademacher ±1 perturbations,
            # std-normalized projected gradient, then ONE more baseline forward at theta.
            # Update applies (lr * proj_grad[i] * z_i) per sample.
            N = int(getattr(args, "fzoo_n", 8))
            perturbTimes = N // 2 if getattr(self, "_fzoo_losses", None) is not None else N
            self._fzoo_losses = []  # mark for subsequent steps
            self._fzoo_random_seeds = []
            with torch.no_grad():
                for i in range(perturbTimes):
                    seed = int(np.random.randint(1000000000))
                    self._fzoo_random_seeds.append(seed)
                    with _pf("fzoo_perturb_pos"):
                        torch.manual_seed(seed)
                        for name, param in self.named_parameters_to_optim:
                            z = torch.randint(0, 2, size=param.data.size(),
                                              device=param.data.device, dtype=param.data.dtype) * 2 - 1
                            param.data.add_(z, alpha=args.zo_eps)
                    with _pf("fzoo_forward_perturbed"):
                        loss_i = self.zo_forward(model, inputs).detach()
                    self._fzoo_losses.append(loss_i)
                    with _pf("fzoo_restore"):
                        torch.manual_seed(seed)
                        for name, param in self.named_parameters_to_optim:
                            z = torch.randint(0, 2, size=param.data.size(),
                                              device=param.data.device, dtype=param.data.dtype) * 2 - 1
                            param.data.add_(z, alpha=-args.zo_eps)

                loss_stack = torch.stack(self._fzoo_losses).float()
                std = torch.std(loss_stack, unbiased=False).clamp(min=1e-8)
                with _pf("fzoo_forward_baseline"):
                    loss_baseline = self.zo_forward(model, inputs).detach()

            self._fzoo_proj_grads = ((loss_stack - loss_baseline.float()) / (perturbTimes * std)).cpu().tolist()
            avg_loss = loss_baseline.item()
            self.projected_grad = None
        else:
            # Standard MeZO: two-sided estimation with simple averaging
            total_projected_grad = 0.0
            total_loss = 0.0

            for sample_idx in range(group_size):
                self.zo_random_seed = self.zo_random_seeds[sample_idx]

                # First function evaluation (theta + z)
                with _pf("mezo_perturb_pos"):
                    self.zo_perturb_parameters(scaling_factor=1)
                with _pf("mezo_forward_pos"):
                    loss1 = self.zo_forward(model, inputs)

                # Second function evaluation (theta - z)
                with _pf("mezo_perturb_neg"):
                    self.zo_perturb_parameters(scaling_factor=-2)
                with _pf("mezo_forward_neg"):
                    loss2 = self.zo_forward(model, inputs)

                with _pf("mezo_loss_diff"):
                    total_projected_grad += ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                    total_loss += loss1.item()

                # Reset model back to its parameters at start of step
                with _pf("mezo_restore"):
                    self.zo_perturb_parameters(scaling_factor=1)

            # Average the gradient estimates
            self.projected_grad = total_projected_grad / group_size
            avg_loss = total_loss / group_size

        # Note: Gradient accumulation is now supported for zo_optimizer methods
        # For Flipout, gradient accumulation is handled separately in the training loop
        if self.args.gradient_accumulation_steps > 1 and getattr(self.args, "zo_optimizer", "mezo") != "flipout":
            # For non-Flipout ZO methods, gradient accumulation is not yet implemented
            logger.warning("Gradient accumulation > 1 is only supported for Flipout optimizer")
        
        return torch.tensor(avg_loss, device=model.device)


    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients (averaged over multiple samples).
        """
        global _PROFILE_STEP
        args = self.args
        zo_opt = getattr(args, "zo_optimizer", "mezo")
        # FZOO has a different update structure (multi-seed, std-normalized projected grad)
        if zo_opt == "fzoo":
            with _pf("fzoo_update"):
                lr = self._get_learning_rate()
                for idx, seed in enumerate(self._fzoo_random_seeds):
                    torch.manual_seed(seed)
                    proj = self._fzoo_proj_grads[idx]
                    for name, param in self.named_parameters_to_optim:
                        z = torch.randint(0, 2, size=param.data.size(),
                                          device=param.data.device, dtype=param.data.dtype) * 2 - 1
                        param.data.add_(z, alpha=-lr * proj)
                self.lr_scheduler.step()
            _PROFILE_STEP += 1
            if _PROFILE_OUT and len(_PROFILE_DATA) % 100 == 0 and len(_PROFILE_DATA) > 0:
                _pf_flush()
            return

        _zo_update_outer_pf = _pf("mezo_update")
        _zo_update_outer_pf.__enter__()
        group_size = getattr(args, "zo_group_size", 1)

        # Standard MeZO: projected_grad * averaged z
        for name, param in self.named_parameters_to_optim:
            z_sum = torch.zeros_like(param.data)
            for seed in self.zo_random_seeds:
                torch.manual_seed(seed)
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = self._baseline_modify_z(name, param, z)
                z_sum += z
            z_avg = z_sum / group_size

            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z_avg + args.weight_decay * param.data)
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z_avg)

        self.lr_scheduler.step()
        _zo_update_outer_pf.__exit__(None, None, None)
        _PROFILE_STEP += 1  # mezo path completes one optimization step
        # Periodic flush every 100 records
        if _PROFILE_OUT and len(_PROFILE_DATA) % 100 == 0 and len(_PROFILE_DATA) > 0:
            _pf_flush()


    ############## Misc overload functions ##############


    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in getattr(self.args, "sharded_ddp", [])
            or ShardedDDPOption.ZERO_DP_3 in getattr(self.args, "sharded_ddp", [])
            or getattr(self, "fsdp", None) is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
