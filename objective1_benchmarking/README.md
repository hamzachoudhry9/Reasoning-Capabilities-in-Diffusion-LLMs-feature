# Objective 1: Compute-Parity Benchmarking

This folder contains the team's evaluation code for benchmarking LLaDA-8B-Instruct
(diffusion) against Llama-3-8B-Instruct (autoregressive) on math and logical-reasoning
tasks under matched inference compute. Dataset loaders and parsers are imported from
`../d1/eval/` via a sys.path shim; only team-modified or team-added files live here.

---

## Files

| File | Purpose |
|---|---|
| `eval_sft_task2.py` | Main eval harness. Resume-safe JSONL streaming, OOM fallbacks, multi-dataset. Used by `run_eval_all.sh`. |
| `eval.py` | Simpler DDP eval entry (no resume logic). Used by `run_eval.sh`. |
| `eval_gpqa_arrow.py` | GPQA-only eval that reads a local Arrow dataset directory. Same resume and OOM logic as `eval_sft_task2.py`. |
| `generate.py` | Modified LLaDA sampler. Adds proper Gumbel-Max sampling, top-k/top-p filtering, and repetition penalty on top of the d1 upstream. |
| `parse_boxed_accuracy.py` | Post-hoc scoring script. Reads `*_generations.json` files and computes numeric (boxed) and MCQ (letter) accuracy. Aggregates by dataset and variant. |
| `llada_parity_metrics_verbose.py` | Loads `telemetry.rank*.jsonl` files and computes compute-parity and inference-parity pairs between AR and LLaDA settings. Outputs CSVs and a summary JSON. |
| `run_llada_metrics_on_jsonl.py` | Runs LLaDA at a configurable set of step budgets (default: 16, 32, 64, 256) over a `val.jsonl` dataset, appending per-example telemetry. Auto-resumes from existing telemetry. |
| `logiqa.py` | Dataset wrapper for `lucasmccabe/logiqa` (MCQ, A-D choices). |
| `aime2025.py` | Dataset wrapper for AIME 2025 (falls back to AIME 2024 if unavailable). |
| `gpqa_diamond.py` | Dataset wrapper for GPQA diamond/main loaded from HF mirrors, with decontamination against the SFT train.jsonl. |
| `gpqa_diamond_2.py` | Same as `gpqa_diamond.py` but adds support for local Arrow eval sets (`--eval_arrow`). Used by `eval_gpqa_arrow.py`. |
| `run_eval.sh` | Simple DDP sweep over countdown, sudoku, math, gsm8k at gen_length 128 and 256. |
| `run_eval_all.sh` | Resume-safe step-budget sweep using `eval_sft_task2.py`. Configurable datasets, steps, retry count. Writes DONE markers. |
| `run_eval_all_gpqa.sh` | Same as `run_eval_all.sh` but targets GPQA with a local Arrow dataset path. |

---

## Architecture

### Two-tier eval harness

`eval.py` is the simpler script: it wraps `generate()`, collects outputs, and saves
a bundled JSON. It runs under DDP via `torchrun`. Use it for quick sweeps where you
do not need crash recovery or resume support.

`eval_sft_task2.py` is the main harness used for all reported results. Key differences:

- **Streaming JSONL output.** Every batch is appended to `generations.rank{rank}.jsonl`
  immediately after generation (with `fsync`). A `state.json` tracks the next expected
  example index. On restart, the script reads the JSONL line count and `state.json`
  and skips already-processed examples without re-running them.
- **OOM fallbacks.** On `torch.cuda.OutOfMemoryError`, the script optionally retries
  with fp16 casting, 4-bit quantization (via `bitsandbytes`), reduced micro-batches,
  or reduced generation length. All fallbacks are opt-in via CLI flags.
- **Final bundle.** After the loop completes, all JSONL lines are collected and written
  to a single `*_generations.json` for downstream scoring.

### generate.py

The local `generate.py` shadows the d1 upstream because the script directory is
`sys.path[0]`. Additions over the d1 version:

- `gumbel_max_sample()`: correct Gumbel-Max sampling (`argmax(logits/T + Gumbel(0,1))`).
  At `temperature=0` this reduces to greedy argmax. The d1 version uses an approximation.
- `apply_top_k_top_p()`: optional top-k and top-p logit filtering.
- `repetition_penalize_logits()`: optional repetition penalty (divides logits of
  already-seen tokens by a penalty factor).

All additions are off by default (`temperature=0`, `top_k=0`, `top_p=0`, `rep_penalty=1`).

### Scoring

Generation and accuracy scoring are decoupled. The eval scripts save raw decoded
text; `parse_boxed_accuracy.py` runs afterward and scores two answer formats:

- **Numeric**: extracts `\boxed{...}` content and compares to the numeric ground truth.
- **MCQ letter**: extracts A/B/C/D from `\boxed{A}`, `<answer>` tags, "Final: A"
  patterns, or isolated last-line letters. Used for LogiQA and GPQA.

---

## Setup

Each entry script adds `../d1/eval/` to `sys.path` at startup, so dataset modules
resolve correctly regardless of which directory you run from.

```bash
# Install dependencies (or use d1/env.yml):
pip install torch transformers peft datasets accelerate tqdm numpy pandas pyarrow

# Optional (for 4-bit fallback):
pip install bitsandbytes
```

---

## Running

### Standard sweep (countdown, sudoku, math, gsm8k)

```bash
cd objective1_benchmarking

# DDP sweep at gen_length 128 and 256, edit GPU ids as needed:
bash run_eval.sh 0 1 2 3
```

Outputs go to `eval_results/`.

### Resume-safe step-budget sweep

Edit the `USER SETTINGS` block at the top of `run_eval_all.sh` (datasets, steps,
SFT checkpoint path), then:

```bash
bash run_eval_all.sh                   # runs, resumes from JSONL if interrupted
FORCE=1 bash run_eval_all.sh           # re-run even if DONE markers are present
START_INDEX=520 bash run_eval_all.sh   # hard override: resume from example 520
```

Outputs go to `../results/eval_sweep_4datasets/{base,sft}/<dataset>/g<GENLEN>_s<STEPS>/`.
Each run directory contains:

```
generations.rank0.jsonl          streaming output (appended per batch)
state.json                       next example index (for resume)
*_generations.json               final bundled output (written on clean completion)
DONE                             marker written on success
run.log                          full stdout/stderr log
```

### GPQA sweep (Arrow dataset)

Set `GPQA_ARROW_DIR` and `SFT_CKPT` at the top of `run_eval_all_gpqa.sh`, then:

```bash
bash run_eval_all_gpqa.sh
FORCE=1 bash run_eval_all_gpqa.sh
```

Outputs go to `../results/gpqa_eval/`.

### Single-run examples

```bash
# Single config, single GPU, base model, gsm8k:
python eval_sft_task2.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --dataset gsm8k \
  --gen_length 256 --diffusion_steps 64 --block_length 32 \
  --batch_size 4 --output_dir results/gsm8k_base_64steps

# SFT model:
python eval_sft_task2.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --checkpoint_path ../sft_checkpoints/llada_mix_temp07/checkpoint-942 \
  --dataset logiqa \
  --gen_length 256 --diffusion_steps 256 --block_length 32 \
  --batch_size 1 --output_dir results/logiqa_sft_256steps

# GPQA (local Arrow):
python eval_gpqa_arrow.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --checkpoint_path ../sft_checkpoints/llada_mix_temp07/checkpoint-942 \
  --eval_arrow /path/to/diamond_gpqa_test \
  --gen_length 256 --diffusion_steps 64 \
  --batch_size 1 --output_dir results/gpqa_sft_64steps

# With OOM fallbacks enabled:
python eval_sft_task2.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --dataset gpqa_diamond \
  --gen_length 128 --diffusion_steps 16 --batch_size 1 \
  --enable_low_mem_fallback --fp16_fallback --clear_cache_every 1 \
  --output_dir results/gpqa_lowmem
```

### CLI reference for `eval_sft_task2.py`

| Flag | Default | Description |
|---|---|---|
| `--model_path` | `GSAI-ML/LLaDA-8B-Instruct` | HF id or local path of the base model |
| `--checkpoint_path` | `""` | PEFT LoRA adapter id or path (omit for base model) |
| `--dataset` | `gsm8k` | `gsm8k`, `math`, `countdown`, `sudoku`, `logiqa`, `gpqa`, `gpqa_diamond`, `aime2025` |
| `--gen_length` | `128` | Number of tokens to generate |
| `--diffusion_steps` | `gen_length // 2` | Total denoising steps |
| `--block_length` | `32` | Block size for semi-autoregressive decoding |
| `--batch_size` | `4` | Batch size per GPU |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--output_dir` | `results/` | Where to write JSONL and JSON outputs |
| `--resume_state` | `output_dir/state.json` | Path to state file for resume tracking |
| `--start_index` | `-1` | Hard override: resume from this example index |
| `--no_finalize` | off | Skip writing the final `*_generations.json` bundle |
| `--enable_low_mem_fallback` | off | Enable OOM fallback handling |
| `--fp16_fallback` | off | Cast model to fp16 on OOM |
| `--bnb_4bit_fallback` | off | Reload model in 4-bit (requires bitsandbytes) |
| `--clear_cache_every` | `0` | Clear CUDA cache every N batches (0 = disabled) |
| `--print_mem_stats` | off | Print GPU memory stats after each cache clear |

---

## Scoring

After generation, run `parse_boxed_accuracy.py` to compute accuracy:

```bash
# Score all generation files under a directory:
python parse_boxed_accuracy.py --in ../results/eval_sweep_4datasets/

# Score a single file:
python parse_boxed_accuracy.py --in results/gsm8k_base_64steps/gsm8k_instruct_256_128_0_generations.json

# Filter to specific datasets only:
python parse_boxed_accuracy.py \
  --in ../results/ \
  --datasets gsm8k,logiqa,gpqa_diamond
```

Output prints per-file accuracy, per-dataset summary, and a grand aggregate.

---

## Compute-parity and telemetry metrics

`run_llada_metrics_on_jsonl.py` collects latency, GPU-seconds, and peak memory for
LLaDA at multiple step budgets and writes them to `telemetry.rank0.jsonl`. Each
example gets a stable `example_key` so the script can skip already-completed pairs
on restart without re-running them.

```bash
python run_llada_metrics_on_jsonl.py \
  --dataset_jsonl /path/to/val.jsonl \
  --filter_source logiqa \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --checkpoint_path ../sft_checkpoints/llada_mix_temp07/checkpoint-942 \
  --steps_list "16,32,64,256" \
  --gen_length 256 \
  --output_dir results/logiqa_telemetry

# Force re-run from scratch:
python run_llada_metrics_on_jsonl.py ... --force 1
```

`llada_parity_metrics_verbose.py` reads the telemetry files and computes
compute-parity and inference-parity pairs between AR and LLaDA:

```bash
python llada_parity_metrics_verbose.py \
  --root results/logiqa_telemetry \
  --out_dir results/parity_analysis
```

Outputs:

```
per_example_<ts>.csv                  one row per (example, steps)
aggregated_by_family_setting_dataset_<ts>.csv
compute_parity_pairs_<ts>.csv         LLaDA settings matched to AR by GPU-seconds
inference_parity_pairs_<ts>.csv       LLaDA settings matched to AR by p50 latency
pairs_verbose_<ts>.csv                all (AR, LLaDA) candidate pairs with distances
summary_<ts>.json
```

---

## Datasets

| Dataset | Format | Source | Loaded from |
|---|---|---|---|
| GSM8K | Numeric, `\boxed{}` | HF `openai/gsm8k` | `d1/eval/gsm8k.py` |
| MATH-500 | Numeric, `\boxed{}` | HF | `d1/eval/math500.py` |
| Countdown | Numeric | Local (d1/dataset.zip) | `d1/eval/countdown.py` |
| Sudoku | Structured | Local (d1/dataset.zip) | `d1/eval/sudoku.py` |
| LogiQA | MCQ A-D | HF `lucasmccabe/logiqa` | `logiqa.py` (this folder) |
| AIME 2025 | Numeric integer | HF `HuggingFaceH4/aime_2025` | `aime2025.py` (this folder) |
| GPQA Diamond | MCQ A-D | HF mirrors (Idavidrein/gpqa, etc.) | `gpqa_diamond.py` (this folder) |
| GPQA (local Arrow) | MCQ A-D | Local Arrow directory | `gpqa_diamond_2.py` (this folder) |

Countdown and Sudoku test sets are bundled as `d1/dataset.zip`. Extract before
running if the dataset loaders cannot find them.
