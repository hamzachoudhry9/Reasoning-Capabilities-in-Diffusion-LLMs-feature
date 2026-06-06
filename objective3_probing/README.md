# Objective 3: Mechanistic Probing of Diffusion Trajectories

This folder contains the trajectory-level mechanistic probing work. The scripts collect and visualize how the diffusion denoising process behaves internally on correct vs incorrect reasoning chains, comparing a BASE model against an SFT (LoRA fine-tuned) variant.

The attention probing component (Bidirectionality Index, ALI, attention entropy) lives in `d1/objective3/` and is imported via `attention_probes.py`.

---

## Files

| File | Purpose |
|---|---|
| `generate.py` | Extended sampler with three entry points: `generate` (plain decoding), `generate_with_entropy` (logs entropy and flip counts per step), `generate_with_fli` (additionally logs per-position flip masks for FLI and stability). |
| `entropy_eval.py` | Runs `generate_with_entropy` over a dataset, splits results by correctness, and saves mean entropy and flip trajectories to JSON. |
| `fli_eval.py` | Runs `generate_with_fli` over a dataset, computes FLI and stability CDFs per example, and saves aggregated results to JSON. |
| `eval.py` | Standard generation eval compatible with DDP. Saves raw model outputs for downstream parsing. |
| `analysis.ipynb` | Loads JSON outputs from both eval scripts and reproduces all figures: entropy curves, token flip counts, FLI trajectories, stability CDFs, and step-position flip heatmaps. |
| `attention_probes.py` | Imports the attention probe package from `d1/objective3/probes/`. |

---

## Metrics

### Notation

| Symbol | Meaning |
|---|---|
| $s$ | Diffusion step index, $s \in \{1, \dots, S\}$ |
| $i, j$ | Token position in the generated region $G$ |
| $v$ | Vocabulary index |
| $z_{s,i} \in \mathbb{R}^{|V|}$ | Logit vector at step $s$, position $i$ |
| $G$ | Number of generated token positions (= `gen_length`) |

---

### Token Entropy

At each diffusion step $s$, the model outputs a logit vector $z_{s,i}$ for every position $i$. We convert to a probability distribution and compute the Shannon entropy:

$$p_{s,i}(v) = \frac{\exp(z_{s,i}(v))}{\sum_{v'} \exp(z_{s,i}(v'))}$$

$$H_{s,i} = -\sum_{v} p_{s,i}(v) \log p_{s,i}(v)$$

To get one number per sequence per step, we average over the generated region only (the prompt is excluded):

$$H_s = \frac{1}{|G|} \sum_{i \in G} H_{s,i}$$

Plotted curves are group means of $H_s$ over examples, split by model (BASE / SFT) and correctness.

---

### Prediction Flips

At each step $s$, the model's best-guess token at position $i$ is:

$$\hat{x}_{s,i} = \arg\max_{v} \, z_{s,i}(v)$$

A **flip** at position $i$ between steps $s-1$ and $s$ is:

$$f_i(s) = \mathbf{1}\left[\hat{x}_{s,i} \neq \hat{x}_{s-1,\, i}\right]$$

The total flip count across the generated region at step $s$ is:

$$F_s = \sum_{i \in G} f_i(s)$$

High $F_s$ means large portions of the candidate answer are still being rewritten. A falling $F_s$ signals that the trajectory is settling.

---

### Flip Localization Index (FLI)

Flip count alone does not say *where* edits are happening. FLI captures whether the flips at step $s$ are concentrated on a few positions or scattered across the whole answer.

**Step 1.** Treat the per-position flip indicators as a distribution over positions:

$$p_{s,j} = \frac{f_j(s)}{\sum_{k \in G} f_k(s) + \varepsilon}$$

**Step 2.** Compute the positional entropy of that distribution:

$$H_s^{\text{pos}} = -\sum_{j \in G} p_{s,j} \log p_{s,j}$$

**Step 3.** Normalize by the maximum possible entropy $\log G$ and invert:

$$\text{FLI}_s = 1 - \frac{H_s^{\text{pos}}}{\log G}$$

**Interpretation:**

| Value | Meaning |
|---|---|
| $\text{FLI}_s = 0$ | Flips are spread uniformly across all positions (maximally diffuse editing) |
| $\text{FLI}_s = 1$ | All flips land on a single position (perfectly localized editing) |

---

### Stability CDF

FLI tells us where edits are concentrated at each step, but not when individual positions stop changing altogether. The stability curve tracks that.

For each position $j$, define its **last-flip time** as the latest step at which it changed:

$$\ell_j = \max\lbrace s : f_j(s) = 1 \rbrace$$

If a position never flips, set $\ell_j = -1$.

The **stability at step $s$** is the fraction of positions whose last flip occurred at or before $s$:

$$\text{Stab}(s) = \frac{1}{G} \left| \lbrace j : \ell_j \leq s \rbrace \right|$$

This is the empirical CDF of token freeze times across the generated region. A curve that rises steeply in the early steps means the model commits to most of its answer quickly and only refines a small tail afterward.

---

## Running

Run all scripts from inside this folder. Dataset loaders (`gsm8k.py`, `math500.py`, `countdown.py`, `sudoku.py`) are imported from `d1/eval/` via a `sys.path` shim at the top of each script.

**Collect entropy and flip trajectories:**

```bash
# BASE model
python entropy_eval.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --dataset gsm8k \
  --gen_length 256 --diffusion_steps 64 --block_length 32 \
  --max_examples 200 \
  --output_path results/gsm8k_base_entropy.json

# SFT model
python entropy_eval.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --checkpoint_path <lora_path_or_hf_id> \
  --dataset gsm8k \
  --gen_length 256 --diffusion_steps 64 --block_length 32 \
  --max_examples 200 \
  --output_path results/gsm8k_sft_entropy.json
```

**Collect FLI and stability:**

```bash
# BASE model
python fli_eval.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --dataset gsm8k \
  --gen_length 256 --diffusion_steps 64 \
  --max_examples 200 \
  --output_path results/gsm8k_base_fli.json

# SFT model
python fli_eval.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --checkpoint_path <lora_path_or_hf_id> \
  --dataset gsm8k \
  --gen_length 256 --diffusion_steps 64 \
  --max_examples 200 \
  --output_path results/gsm8k_sft_fli.json
```

Then open `analysis.ipynb` and point the four `load_json` calls at these output paths to reproduce all plots.

**Supported datasets:** `gsm8k`, `math`, `countdown`, `sudoku`

---

## Key Findings (GSM8K, 200 examples, gen_length=256, steps=64)

**Entropy:** BASE entropy starts around 3.6-3.9 nats and decays slowly across all 64 steps with no clear separation between correct and wrong examples. SFT entropy starts lower at roughly 2.1-2.7 nats and collapses much faster on correct examples, reaching near-zero by around step 30.

**Flips:** BASE produces a burst of high flip counts in the earliest steps and continues rewriting dozens of tokens per step through the midpoint of the schedule. SFT-correct drops below roughly 10 flips per step within the first 15-20 steps and reaches only a handful by the end.

**FLI:** BASE FLI grows to around 0.6-0.7 with no consistent separation between correct and wrong trajectories. SFT-correct FLI climbs to roughly 0.9 in the final steps, meaning nearly all remaining edits are focused on a narrow band of positions. SFT-wrong FLI falls between the two.

**Stability:** BASE tokens freeze at a roughly linear rate regardless of outcome. For SFT-correct, nearly half of generated tokens are already stable after only a few steps, and the curve reaches around 0.8-0.9 by the midpoint.

**Flip heatmaps:** BASE shows a broad triangular band of editing activity across both the position and step axes. The SFT heatmap has a compact wedge near the start of the generated region in the early-to-middle steps; the rest of the sequence fades to near-zero flip probability well before the schedule ends.

Taken together, these results show that SFT converts diffusion decoding from a process that rewrites the answer globally into one that commits early and then focuses its remaining edit budget on a narrow set of positions that are likely critical for correctness.
