# entropy_eval.py
import argparse, json, re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from generate import generate_with_entropy

# Datasets shipped with the repo
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


# ---------- helpers for numeric correctness ----------
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
        m = ANY_NUM.findall(str(gt))
        gt_val = float(m[-1]) if m else None
    return (gt_val is not None) and abs(pred - gt_val) < 1e-6
# ----------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True,
                    help="HF id or local path of BASE model, e.g. GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--checkpoint_path", type=str, default="",
                    help="Optional PEFT LoRA id/path for SFT probing")
    ap.add_argument("--dataset", choices=list(DATASET_MAP.keys()), default="gsm8k")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--diffusion_steps", type=int, default=64)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--max_examples", type=int, default=100,
                    help="Subsample size for the eval dataset (0 = repo default)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    ap.add_argument("--mask_id", type=int, default=126336, help="Model's MASK token id (LLaDA uses 126336)")
    ap.add_argument("--output_path", type=str, default="entropy_flips.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- load tokenizer + model (+ optional LoRA) -----
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)

    if args.checkpoint_path:
        # Attach LoRA (SFT)
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    # ----- dataset -----
    Dataset = DATASET_MAP[args.dataset]
    dataset = Dataset(tokenizer, num_examples=0, add_reasoning=True, subsample=args.max_examples)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # ----- accumulators -----
    correct_ent_trajs, wrong_ent_trajs = [], []
    correct_flip_trajs, wrong_flip_trajs = [], []

    # ----- loop -----
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)   # [B, Lprompt]
            gts = batch["answers"]                      # list[str|float]

            out, entropy_traj, flip_traj = generate_with_entropy(
                model,
                input_ids,
                tokenizer=None,
                steps=args.diffusion_steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                mask_id=args.mask_id,
            )

            # decode only generated tail for correctness check
            gen_texts = tokenizer.batch_decode(out[:, -args.gen_length:], skip_special_tokens=False)

            # numpy-ify
            ent_np  = entropy_traj.cpu().numpy()  # [S, B]
            flip_np = flip_traj.cpu().numpy()     # [S, B]
            S, B = ent_np.shape

            # split by correctness
            for b in range(B):
                ok = is_correct_numeric(gen_texts[b], gts[b])
                ent_b  = ent_np[:, b].tolist()
                flip_b = flip_np[:, b].tolist()
                if ok:
                    correct_ent_trajs.append(ent_b)
                    correct_flip_trajs.append(flip_b)
                else:
                    wrong_ent_trajs.append(ent_b)
                    wrong_flip_trajs.append(flip_b)

    # ----- summarise & save -----
    def mean_or_none(trajs):
        return None if not trajs else np.mean(np.array(trajs), axis=0).tolist()

    res = {
        "dataset": args.dataset,
        "gen_length": args.gen_length,
        "diffusion_steps": args.diffusion_steps,
        "num_correct": len(correct_ent_trajs),
        "num_wrong": len(wrong_ent_trajs),

        # entropy
        "mean_entropy_correct": mean_or_none(correct_ent_trajs),
        "mean_entropy_wrong":   mean_or_none(wrong_ent_trajs),
        "all_entropy_correct_trajs": correct_ent_trajs,
        "all_entropy_wrong_trajs":   wrong_ent_trajs,

        # prediction flips
        "mean_flips_correct": mean_or_none(correct_flip_trajs),
        "mean_flips_wrong":   mean_or_none(wrong_flip_trajs),
        "all_flips_correct_trajs": correct_flip_trajs,
        "all_flips_wrong_trajs":   wrong_flip_trajs,
    }
    with open(args.output_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.output_path}")
    print(f"  correct={len(correct_ent_trajs)}  wrong={len(wrong_ent_trajs)}")


if __name__ == "__main__":
    main()
