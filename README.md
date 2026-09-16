# GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning

**Findings of EMNLP 2026**

Liyan Tan, Yequan Zhao, Yifan Yang, Ruijie Zhang, Xinling Yu, Zheng Zhang

[[arXiv](https://arxiv.org/abs/2606.02857)] [[PDF](https://arxiv.org/pdf/2606.02857)] [[Project page](https://liyantan111.github.io/papers/grzo/)]

One perturbation per example instead of one per batch. GRZO raises the number of
zeroth-order gradient directions from one to the batch size at no extra forward cost,
beating MeZO by +3.0 average accuracy while staying within 0.5% of the forward-only
inference memory floor. It is a drop-in replacement for MeZO and composes with its
sparse, low-rank and quantized variants, lifting them by +4.9 on average.

![Side-by-side pipeline comparison of MeZO and GRZO](https://liyantan111.github.io/images/papers/grzo-pipeline.png)

MeZO (left) shares one perturbation across the mini-batch, giving a single gradient
direction per step. GRZO (right) builds pseudo-independent per-example perturbations and
applies group-relative normalization, yielding B effective directions and much lower
variance at the same forward budget.

**Problem.** Zeroth-order fine-tuning removes backpropagation's memory cost, but one
perturbation shared across the mini-batch makes the gradient estimate too noisy to match
first-order training. That variance is what keeps ZO methods from closing the accuracy gap.

**Method.** GRZO gives every example its own pseudo-independent perturbation and combines
the per-example losses by group-relative normalization: B gradient directions per step
instead of one, at the same forward cost and with peak memory still at the inference floor.

**Main result.** Highest accuracy among ZO methods at inference-level memory: 81.6% against
MeZO's 74.4% on RTE (Llama3-8B), +3.0 on average, at 17.82 GB peak memory, 0.5% above the
forward-only floor. The cost is a 23% per-step time premium, which it repays by converging
fastest in wall-clock time. Dropped into sparse, low-rank, and quantized ZO variants it
lifts them by +4.9 on average.

## How it works

For a linear layer $W \in \mathbb{R}^{d_{out}\times d_{in}}$ and a batch of $B$ examples,
GRZO draws one shared base noise $U \sim \mathcal{N}(0, I)$ and per-example Rademacher
sign vectors $r_i \in \{\pm1\}^{d_{out}}$, $s_i \in \{\pm1\}^{d_{in}}$. Example $i$ sees
the perturbation $\Delta_i = U \odot (r_i s_i^{\top})$. A forward hook adds
$\sigma\, r_i \odot \big(U (s_i \odot x_i)\big)$ to the output of example $i$, so one
batched forward evaluates all $B$ perturbed models without materializing $B$ weight
copies. Two forwards ($\pm\sigma$) give per-example losses $\ell_i^{\pm}$, and

$$\delta_i = \tfrac{1}{2}\,(\ell_i^{+} - \ell_i^{-}), \qquad
a_i = \frac{\delta_i}{\operatorname{std}_j(\delta_j) + \varepsilon}, \qquad
\hat g_W = \frac{1}{2\sigma B}\sum_i a_i \Delta_i
         = \frac{U \odot \big(R^{\top}\operatorname{diag}(a)\,S\big)}{2\sigma B}.$$

The update $W \leftarrow W - \eta\,\hat g_W$ costs one $d_{out}\times B\times d_{in}$
matmul per layer; $U$ is regenerated from a seed and never stored. Norm layers use
per-example $\pm1$ vectors and embeddings receive sparse row updates for the tokens in the
batch. Under data parallelism (`grzo_flipout_step_ddp`) the seed of $U$ is broadcast, the
signs are per-rank, the losses are all-gathered so the advantage is computed over the
global batch, and only the small response matrix $R^{\top}\operatorname{diag}(a)S$ is
all-reduced.

## Results

![Per-step time, peak memory and accuracy for MeZO, S-MeZO, LOZO and GRZO](https://liyantan111.github.io/images/papers/grzo-efficiency.png)

GRZO at a glance on RTE (Llama3-8B). Left: the highest accuracy (81.6%) at inference-level
peak memory (17.82 GB, 0.5% over the forward-only floor), for a 23% per-step time premium
over MeZO. Right: the fastest convergence in both training steps and wall-clock time.

![Training loss curves on four tasks against steps and wall-clock time](https://liyantan111.github.io/images/papers/grzo-alltasks.png)

Training-loss curves on Llama3-8B (RTE, MultiRC) and OPT-13B (SQuAD, DROP), plotted
against both training steps and wall-clock time. The per-step premium is repaid: GRZO
reaches any given loss level sooner on the clock.

| Method | SST-2 | RTE | CB | BoolQ | WiC | MultiRC | COPA | SQuAD (F1) | DROP (F1) |
|---|---|---|---|---|---|---|---|---|---|
| **Llama3-8B** | | | | | | | | | |
| Adam (FO) | 96.0 | 92.0 | 92.0 | 86.6 | 72.6 | 84.7 | 89.0 | 90.4 | 59.4 |
| LoRA (FO) | 95.0 | 80.9 | 73.2 | 86.4 | 70.7 | 82.4 | 89.0 | 89.4 | 58.2 |
| MeZO | 92.2 | 74.4 | 69.6 | 76.7 | 57.8 | 77.6 | 88.0 | **86.7** | 57.1 |
| FZOO | 93.0 | 76.6 | 68.6 | 81.2 | 59.4 | 77.6 | 89.0 | 86.0 | 57.4 |
| GRZO (ours) | **93.4** | **81.6** | **72.0** | **81.4** | **59.8** | **78.6** | **89.0** | 86.2 | **65.0** |
| **OPT-13B** | | | | | | | | | |
| Adam (FO) | 95.3 | 80.9 | 94.6 | 83.5 | 66.3 | 76.2 | 88.0 | 89.5 | 31.3 |
| LoRA (FO) | 94.8 | 78.3 | 69.6 | 80.2 | 64.3 | 69.4 | 89.0 | 88.0 | 30.9 |
| MeZO | 91.4 | 66.1 | 66.0 | 67.6 | **59.4** | 57.3 | **88.0** | 84.7 | 30.9 |
| FZOO | **93.8** | 76.8 | 69.6 | **72.2** | **59.4** | 57.6 | 87.0 | 84.8 | 28.7 |
| GRZO (ours) | 93.4 | **78.0** | **70.2** | 70.4 | 58.6 | **57.8** | **88.0** | **85.2** | **32.8** |

Main results. Adam and LoRA are first-order methods and need full backpropagation memory.
Among ZO methods, the best number per task is in bold.

| Method | BoolQ | RTE | COPA | SQuAD (F1) | DROP (F1) |
|---|---|---|---|---|---|
| Sparse-GRZO | 85.1 (+4.6 / +3.7) | 79.4 (+6.0 / −2.2) | 88.0 (+5.0 / −1.0) | 89.0 (+1.5 / +2.8) | 59.3 (+10.9 / −5.7) |
| LO-GRZO | 84.4 (+5.0 / +3.0) | 75.1 (+3.0 / −6.5) | 90.0 (+6.0 / +1.0) | 88.4 (−0.6 / +2.2) | 65.5 (+0.1 / +0.5) |
| Qu-GRZO (int8) | 79.3 (+2.5 / −2.1) | 80.5 (+5.3 / −1.1) | 91.0 (+4.0 / +2.0) | 88.6 (+8.0 / +2.4) | 63.9 (+11.6 / −1.1) |

GRZO as a drop-in replacement for the MeZO core inside orthogonal ZO variants, on
Llama3-8B. The two numbers in each cell are the change against the paired baseline
(Sparse-MeZO, LOZO, QuZO) and against vanilla GRZO (BoolQ 81.4, RTE 81.6, COPA 89.0,
SQuAD 86.2, DROP 65.0). Swapping in GRZO improves every paired baseline on almost every task.

## Code

The training pipeline is derived from [MeZO](https://github.com/princeton-nlp/MeZO)
(`MeZO/large_models/`); GRZO itself lives in `optimizers/flipout.py`.

```
optimizers/flipout.py        GRZO step (single-GPU and DDP) incl. sparse / low-rank / quantized variants
optimizers/quzo_wquant.py    QuZO weight fake-quantization (optional, --quzo_weight_bits)
MeZO/large_models/           run.py (entry point), trainer.py (optimizer dispatch), tasks, templates
scripts/run_llama.sh         launcher for Llama-3-8B on SuperGLUE / QA tasks
scripts/profile.sh           per-step time and peak-memory profiling of every optimizer
scripts/summarize_profile.py summarize the profiling JSON into a table
scripts/profile_breakdown.py synthetic-batch profiler that breaks a GRZO step into stages
```

### Setup

```bash
git clone https://github.com/LiyanTan111/GRZO-public.git && cd GRZO-public
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, torch 2.11, transformers 4.46.3 on 4×A100-40GB. Datasets are
downloaded from the Hugging Face Hub on first use.

### Fine-tune Llama-3-8B

```bash
export MODEL_PATH=/path/to/Meta-Llama-3-8B   # or meta-llama/Meta-Llama-3-8B with HF_TOKEN set

bash scripts/run_llama.sh                                        # GRZO on RTE, 4 GPUs, 20k steps
TASK=BoolQ OPTIMIZER=grzo_sparse      LR=1e-7 bash scripts/run_llama.sh
TASK=SQuAD OPTIMIZER=grzo_lozo_strict LR=1e-7 bash scripts/run_llama.sh
NPROC=1 PER_DEVICE_BATCH=16 bash scripts/run_llama.sh             # single GPU
```

Tasks: `SST2 RTE CB BoolQ WSC WIC MultiRC Copa ReCoRD SQuAD DROP`. Defaults follow the
paper: 1000 train / 500 dev / 1000 eval examples, global batch 16, $\sigma$ = 1e-3,
constant learning rate, fp16 weights, 20k steps, evaluation every 4k steps. Outputs go to
`results/<task>-<optimizer>-lr<lr>-seed<seed>/`. Every setting is an environment variable
(see the header of `scripts/run_llama.sh`); other `run.py` flags pass through `EXTRA="..."`.

| Paper | `--zo_optimizer` | Variant-specific flags |
|---|---|---|
| GRZO | `flipout` | `--estimation_side two_norm` (group-relative; `two` uses the raw $\delta_i$), `--zo_eps`, `--u_distribution` |
| Sparse-GRZO | `grzo_sparse` | `--sparse_ratio 0.25 --sparse_rule small` (keep the smallest-magnitude 25% of each Linear weight active, the Sparse-MeZO rule) |
| LO-GRZO | `grzo_lozo_strict` | `--lozo_rank 8 --lozo_step_interval 50` ($U \leftarrow uv^{\top}/\sqrt r$, $v$ resampled every 50 steps, $s_i \equiv 1$); `grzo_lozo` keeps the per-example $s_i$ |
| Qu-GRZO | `grzo_quzo` | `--quant_bits 8` (two independent stochastic-rounding quantizations of $U$ for the forward and the update); `--quzo_weight_bits 4` also fake-quantizes Linear weights |
| MeZO, Sparse-MeZO, LOZO, QuZO, FZOO | `mezo`, `mezo_sparse`, `mezo_lozo`, `mezo_quzo`, `fzoo` | same flags as the matching GRZO variant; `--fzoo_n 8` |

### Profiling

```bash
bash scripts/profile.sh                      # 25 steps per method on RTE, writes profiling/<method>_rank*.json
python scripts/summarize_profile.py profiling
```

Setting `GRZO_PROFILE_OUT=<file>.json` makes the trainer time every stage of a ZO step
(`torch.cuda.synchronize` + `perf_counter`) and record per-stage peak GPU memory, so any
`scripts/run_llama.sh` invocation can be profiled.

## Citation

```bibtex
@inproceedings{tan2026grzo,
  title     = {GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning},
  author    = {Tan, Liyan and Zhao, Yequan and Yang, Yifan and Zhang, Ruijie and Yu, Xinling and Zhang, Zheng},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## Acknowledgements

Built on [MeZO](https://github.com/princeton-nlp/MeZO) (MIT license, see `MeZO/LICENSE`).
The baseline variants follow the official implementations of
[LOZO](https://github.com/optsuite/LOZO), [Sparse-MeZO](https://github.com/NUS-HPC-AI-Lab/SparseMeZO),
[QuZO](https://github.com/lloo099/QuZO) and [FZOO](https://arxiv.org/abs/2506.09034).
