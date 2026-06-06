# fli_eval.py
import argparse, json, re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from generate import generate_with_fli
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}

BOXED_NUM = re.compile(r"\\boxed\{([-+]?\d+(?:\.\d+)?)\}")
ANY_NUM   = re.compile(r"[-+]?\d+(?:\.\d+)?")

def extract_last_number(text: str):
    m = BOXED_NUM.findall(text)
    if m:
        return float(m[-1])
    m = ANY_NUM.findall(text)
    return float(m[-1]) if m else None

def is_correct_numeric(pred_text: str, gt):
    pred = extract_last_number(pred_text)
    if pred is None:
        return False
    try:
        gt_val = float(gt)
    except Exception:
        gt_val = extract_last_number(str(gt))
    return (gt_val is not None) and abs(pred - gt_val) < 1e-6

def mean_or_none(arrs):
    if not arrs:
        return None
    return np.mean(np.stack(arrs, axis=0), axis=0).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--checkpoint_path", default="")
    ap.add_argument("--dataset", default="gsm8k", choices=list(DATASET_MAP.keys()))
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--diffusion_steps", type=int, default=64)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--max_examples", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    ap.add_argument("--mask_id", type=int, default=126336)
    ap.add_argument("--output_path", default="fli_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    Dataset = DATASET_MAP[args.dataset]
    ds = Dataset(tok, num_examples=0, add_reasoning=True, subsample=args.max_examples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ds.collate_fn)

    corr_pos_masks, wrong_pos_masks = [], []        # each: [S, G] bool
    corr_flip_counts, wrong_flip_counts = [], []    # each: [S]
    corr_fli, wrong_fli = [], []                    # each: [S]
    corr_stab, wrong_stab = [], []                  # each: [S]

    with torch.no_grad():
        for batch in dl:
            inp = batch["input_ids"].to(device)
            gts = batch["answers"]

            out, ent_traj, flips_per_step, flip_pos_mask = generate_with_fli(
                model,
                inp,
                tokenizer=None,
                steps=args.diffusion_steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                mask_id=args.mask_id,
            )

            gen_txt = tok.batch_decode(out[:, -args.gen_length:], skip_special_tokens=False)

            S, B, G = flip_pos_mask.shape
            # sanity: flips_per_step sums == pos flips
            # (optional check, can be removed for speed)
            # assert np.allclose(
            #     flips_per_step.cpu().numpy(),
            #     flip_pos_mask.sum(dim=2).cpu().numpy()
            # )

            flip_pos_np  = flip_pos_mask.cpu().numpy().astype(np.float32)   # [S,B,G]
            flips_step_np = flips_per_step.cpu().numpy().astype(np.float32) # [S,B]

            for b in range(B):
                ok = is_correct_numeric(gen_txt[b], gts[b])

                pos   = flip_pos_np[:, b, :]    # [S, G] 0/1
                flips = flips_step_np[:, b]     # [S]

                # ----- FLI -----
                # For each step s, p_j(s) = normalized flips over positions.
                eps = 1e-9
                denom = pos.sum(axis=1, keepdims=True) + eps   # [S,1]
                p = pos / denom                                # [S,G]
                H = -(p * np.log(p + eps)).sum(axis=1)         # [S]
                H_norm = H / (np.log(G) + eps)                 # [S]
                fli = 1.0 - H_norm                             # [S], 0≈uniform, 1≈localized

                # ----- Stability CDF -----
                # For each position j, last step where it flipped.
                step_ids = np.arange(S, dtype=np.float32)[:, None]   # [S,1]
                # pos is 0/1; for no-flip positions this stays 0
                last_step = (pos * step_ids).max(axis=0)             # [G]
                # A position is "stable by step s" if it never flips after s
                stab = np.array([(last_step <= s).mean() for s in range(S)], dtype=np.float32)  # [S]

                if ok:
                    corr_pos_masks.append(pos)
                    corr_flip_counts.append(flips)
                    corr_fli.append(fli)
                    corr_stab.append(stab)
                else:
                    wrong_pos_masks.append(pos)
                    wrong_flip_counts.append(flips)
                    wrong_fli.append(fli)
                    wrong_stab.append(stab)

    # Choose S,G from collected data (fallback to args if empty)
    if corr_pos_masks:
        S_val, G_val = corr_pos_masks[0].shape
    elif wrong_pos_masks:
        S_val, G_val = wrong_pos_masks[0].shape
    else:
        S_val, G_val = args.diffusion_steps, args.gen_length

    res = {
        "S": S_val,
        "G": G_val,
        # heatmaps
        "mean_posmask_correct": mean_or_none(corr_pos_masks),   # [S,G]
        "mean_posmask_wrong":   mean_or_none(wrong_pos_masks),  # [S,G]
        # flips per step
        "mean_flips_correct": mean_or_none(corr_flip_counts),   # [S]
        "mean_flips_wrong":   mean_or_none(wrong_flip_counts),  # [S]
        # FLI
        "mean_fli_correct": mean_or_none(corr_fli),             # [S]
        "mean_fli_wrong":   mean_or_none(wrong_fli),            # [S]
        # Stability
        "mean_stability_correct": mean_or_none(corr_stab),      # [S]
        "mean_stability_wrong":   mean_or_none(wrong_stab),     # [S]
        # counts
        "num_correct": len(corr_pos_masks),
        "num_wrong":   len(wrong_pos_masks),
    }

    with open(args.output_path, "w") as f:
        json.dump(res, f, indent=2)
    print("[saved]", args.output_path)
    print("  correct:", res["num_correct"], "wrong:", res["num_wrong"])

if __name__ == "__main__":
    main()
