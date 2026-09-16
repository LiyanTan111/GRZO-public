# GRZO: Group-Relative Zeroth-Order Optimization

Official implementation of
**[GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning](https://arxiv.org/abs/2606.02857)**
(Findings of EMNLP 2026).
Liyan Tan, Yequan Zhao, Yifan Yang, Ruijie Zhang, Xinling Yu, Zheng Zhang — University of California, Santa Barbara.

Forward-only (zeroth-order) full-parameter fine-tuning of LLMs at inference-level memory.
GRZO gives **every example in a batch its own perturbation**, evaluates all of them in a
single batched forward pass through a flipout-style factorization, and aggregates the
per-example loss differences with a **group-relative (batch-normalized) weighting**.
It composes with other ZO efficiency methods (low-rank, sparse, and quantized
perturbations) and runs Llama-3-8B on 4×A100-40GB with ~0 GB of optimizer overhead.

The training pipeline is derived from [MeZO](https://github.com/princeton-nlp/MeZO)
(`MeZO/large_models/`); GRZO itself lives in `optimizers/flipout.py`.

## Method

For a linear layer $W \in \mathbb{R}^{d_{out}\times d_{in}}$ and a batch of $B$ examples,
GRZO draws one shared base noise $U \sim \mathcal{N}(0, I)$ and per-example Rademacher
sign vectors $r_i \in \{\pm1\}^{d_{out}}$, $s_i \in \{\pm1\}^{d_{in}}$. Example $i$ sees
the perturbation

$$\Delta_i = U \odot (r_i s_i^{\top}).$$

Instead of materializing $B$ weight copies, a forward hook adds
$\sigma\, r_i \odot \big(U (s_i \odot x_i)\big)$ to the output of example $i$, so one
batched forward evaluates all $B$ perturbed models (the flipout trick). Two forwards
($\pm\sigma$) give per-example losses $\ell_i^{\pm}$ and

$$\delta_i = \tfrac{1}{2}\,(\ell_i^{+} - \ell_i^{-}), \qquad
a_i = \frac{\delta_i}{\operatorname{std}_j(\delta_j) + \varepsilon}
\quad\text{(group-relative advantage)}.$$

The gradient estimate and update are

$$\hat g_W = \frac{1}{2\sigma B}\sum_i a_i \Delta_i
          = \frac{U \odot \big(R^{\top}\operatorname{diag}(a)\,S\big)}{2\sigma B},
\qquad W \leftarrow W - \eta\,\hat g_W,$$

which costs one $d_{out}\times B\times d_{in}$ matmul per layer, and $U$ is never stored
(it is regenerated from a seed). Norm layers use per-example $\pm1$ vectors; embeddings
receive sparse row updates for the tokens present in the batch. Under multi-GPU data
parallelism (`grzo_flipout_step_ddp`) the seed of $U$ is broadcast, the signs are
per-rank, the losses are all-gathered so the advantage is computed over the global
batch, and only the small response matrix $R^{\top}\operatorname{diag}(a)S$ is all-reduced.

### GRZO + X variants

| `--zo_optimizer` | Name | What changes |
|---|---|---|
| `flipout` | **GRZO** | as above |
| `grzo_lozo` | LR-GRZO | $U \leftarrow uv^{\top}/\sqrt{r}$ with $u\in\mathbb{R}^{d_{out}\times r}$ resampled every step and $v\in\mathbb{R}^{r\times d_{in}}$ every `--lozo_step_interval` steps ([LOZO](https://arxiv.org/abs/2410.07698)) |
| `grzo_lozo_strict` | LOZO-GRZO | as `grzo_lozo` with $s_i \equiv 1$, so the update stays in the span of $v$ |
| `grzo_sparse` | Sparse-GRZO | $U \leftarrow U \odot M$; $M$ keeps the `--sparse_ratio` fraction of smallest-magnitude weights (`--sparse_rule small`, the [Sparse-MeZO](https://arxiv.org/abs/2402.15751) rule) or largest (`large`) |
| `grzo_quzo` | QuZO-GRZO | the forward and the update use two independent stochastic-rounding quantizations of $U$ at `--quant_bits` bits, so $\mathbb{E}[u_1 u_2^{\top}] = UU^{\top}$ ([QuZO](https://arxiv.org/abs/2502.12346)); `--quzo_weight_bits 4` additionally fake-quantizes Linear weights in the forward pass |

Baselines run through the same trainer: `mezo`, `mezo_lozo`, `mezo_sparse`, `mezo_quzo`,
and `fzoo` ([FZOO](https://arxiv.org/abs/2506.09034)).

Other GRZO flags: `--zo_eps` ($\sigma$, default 1e-3), `--estimation_side two_norm`
(group-relative; `two` uses the raw $\delta_i$), `--u_distribution gaussian|rademacher`,
and the optional safeguards `--adv_std_floor` / `--adv_clip`.

## Setup

```bash
git clone https://github.com/LiyanTan111/GRZO-public.git && cd GRZO-public
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, torch 2.11, transformers 4.46.3 on 4×A100-40GB.
Datasets are downloaded from the Hugging Face Hub on first use.

## Fine-tune Llama-3-8B

```bash
export MODEL_PATH=/path/to/Meta-Llama-3-8B   # or meta-llama/Meta-Llama-3-8B with HF_TOKEN set

bash scripts/run_llama.sh                                    # GRZO on RTE, 4 GPUs, 20k steps
TASK=BoolQ OPTIMIZER=grzo_lozo  LR=1e-7 bash scripts/run_llama.sh
TASK=SQuAD OPTIMIZER=grzo_sparse LR=1e-7 bash scripts/run_llama.sh
NPROC=1 PER_DEVICE_BATCH=16 bash scripts/run_llama.sh         # single GPU
```

Tasks: `SST2 RTE CB BoolQ WSC WIC MultiRC Copa ReCoRD SQuAD DROP`.
Defaults (paper setting): 1000 train / 500 dev / 1000 eval examples, global batch 16,
$\sigma$ = 1e-3, constant learning rate, fp16 weights, 20k steps, evaluation every 4k
steps. Outputs go to `results/<task>-<optimizer>-lr<lr>-seed<seed>/`.
The launcher exposes every knob as an environment variable (see the header of
`scripts/run_llama.sh`); anything else can be passed through `EXTRA="..."`.

## Profiling

`profiling/REPORT.md` reports per-step wall-clock and peak GPU memory for GRZO, the
three GRZO + X variants, and their MeZO-family counterparts (Llama-3-8B, RTE, global
batch 16, fp16, 4×A100-40GB, 20 measured steps per rank):

| Method | ms / step | peak GB | Method | ms / step | peak GB |
|---|---|---|---|---|---|
| MeZO | 805 | 20.8 | **GRZO** | 973 | **16.0** |
| LOZO | 957 | 20.9 | **LOZO-GRZO** | 990 | **16.1** |
| Sparse-MeZO | 1056 | 27.3 | **Sparse-GRZO** | 1189 | **22.5** |
| QuZO | 2041 | 23.8 | **QuZO-GRZO** | 2566 | **17.3** |
| FZOO | 1949 | – | | | |

Llama-3-8B fp16 weights alone occupy 16.0 GB, so GRZO trains at inference-level memory.
Raw per-rank measurements are in `profiling/time/` and `profiling/memory/`.

To reproduce, set `MODEL_PATH` and run

```bash
bash scripts/profile.sh                                   # ~25 steps per method, writes profiling/new/
python scripts/summarize_profile.py profiling/time profiling/memory
```

The trainer instruments each stage of a ZO step whenever `GRZO_PROFILE_OUT=<file>.json`
is set (`torch.cuda.synchronize` + `perf_counter`, plus per-stage peak memory), so any
`scripts/run_llama.sh` invocation can be profiled. `scripts/profile_breakdown.py` is a
standalone synthetic-batch profiler that breaks a GRZO step down into forward / sign
generation / noise generation / update.

## Acknowledgements

Built on [MeZO](https://github.com/princeton-nlp/MeZO) (MIT license, see `MeZO/LICENSE`).
The baseline variants follow the official implementations of
[LOZO](https://github.com/optsuite/LOZO), [Sparse-MeZO](https://github.com/NUS-HPC-AI-Lab/SparseMeZO),
[QuZO](https://github.com/lloo099/QuZO) and [FZOO](https://arxiv.org/abs/2506.09034).

## Citation

```bibtex
@inproceedings{tan2026grzo,
  title     = {{GRZO}: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning},
  author    = {Tan, Liyan and Zhao, Yequan and Yang, Yifan and Zhang, Ruijie and Yu, Xinling and Zhang, Zheng},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.02857}
}
```
