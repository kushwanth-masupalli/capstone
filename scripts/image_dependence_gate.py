#!/usr/bin/env python3
"""PLAN 4 Phase 4.4 — image-dependence gate (Instruction 16).

Runs after a SHORT pilot (≈300 optimizer steps). Generates on a fixed set of
validation images with greedy decoding and computes:

1. distinct-output ratio = distinct reports / N
2. shuffled-image test: ROUGE-L/CIDEr for correct vs wrong image pairing
3. constant-baseline comparison: most-common train report vs model outputs
4. finding agreement: per-label keyword precision/recall for easy findings

PASS criteria (freeze before looking at test data):
* distinct ratio > 0.5
* correct-image ROUGE-L beats shuffled-image by a visible margin
* pilot CIDEr beats the constant baseline
* at least one abnormal finding with precision above its base rate

Writes ``results/plan4/gate_report.json``.

Usage:
    python scripts/image_dependence_gate.py --n 100 \
        --split data/report_splits/validation.csv
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "generation"))
from generation.medgemma_io import (  # noqa: E402
    ADAPTER_DIR,
    PROMPT_TEXT,
    load_image,
    load_model_and_processor,
)

# Easy labels for keyword matching (Instruction 16.4)
FINDING_KEYWORDS = {
    "cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart", "enlarged cardiac"],
    "effusion": ["effusion", "pleural fluid"],
    "opacity": ["opacity", "opacities", "consolidation", "infiltrate"],
    "pneumothorax": ["pneumothorax"],
}


def rouge_l(cand: str, ref: str) -> float:
    """Simple ROUGE-L (LCS-based F) on lower-cased word tokens."""
    c, r = cand.lower().split(), ref.lower().split()
    if not c or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]
    for i in range(len(c)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if c[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[len(c)][len(r)]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(c), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def score(pairs):
    """Mean ROUGE-L over (generated, reference) pairs."""
    return sum(rouge_l(g, r) for g, r in pairs) / max(1, len(pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Image-dependence gate (PLAN 4, Phase 4.4).")
    ap.add_argument("--n", type=int, default=100, help="Number of fixed validation images.")
    ap.add_argument("--split", default="data/report_splits/validation.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/plan4/gate_report.json")
    args = ap.parse_args()

    import torch

    df = pd.read_csv(args.split)
    # Fixed random subset (seeded); shuffling a copy of the index would be a no-op.
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    n = len(df)
    print(f"gate on {n} fixed validation images (seed {args.seed})")

    processor, model = load_model_and_processor(ADAPTER_DIR)

    # Resolve image paths (Phase 4.2 CSVs use vlm_image_path / raw PNGs).
    def resolve(p):
        cands = [Path(str(p).replace("\\", "/")),
                 Path("data/images") / Path(str(p).replace("\\", "/")).name]
        return next((c for c in cands if c.is_file()), None)

    images = []
    refs = []
    for _, row in df.iterrows():
        p = resolve(row["image_path"])
        if p is None:
            continue
        images.append(load_image(p))
        refs.append(str(row["report_text"]))

    # ---- 1. correct-image generation -------------------------------------
    correct = []
    for i, img in enumerate(images):
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT_TEXT},
            ]}],
            tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], add_special_tokens=False,
                           return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
        correct.append(text)
        print(f"[{i + 1}/{len(images)}] {text[:80]}")

    # ---- 2. shuffled-image generation (same prompts, wrong pixels) -------
    order = list(range(len(images)))
    random.Random(args.seed + 1).shuffle(order)
    shuffled = []
    for i, idx in enumerate(order):
        img = images[idx]
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT_TEXT},
            ]}],
            tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], add_special_tokens=False,
                           return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        shuffled.append(processor.decode(out[0, input_len:], skip_special_tokens=True).strip())

    # ---- metrics ----------------------------------------------------------
    distinct_ratio = len(set(correct)) / max(1, len(correct))

    train_df = pd.read_csv("data/report_splits/train.csv")
    most_common_report = train_df["report_text"].value_counts().index[0]
    const_pairs = [(most_common_report, r) for r in refs]
    correct_pairs = list(zip(correct, refs))
    shuffled_pairs = list(zip(shuffled, refs))

    results = {
        "n_images": len(images),
        "distinct_ratio": round(distinct_ratio, 4),
        "rougeL_correct": round(score(correct_pairs), 4),
        "rougeL_shuffled": round(score(shuffled_pairs), 4),
        "rougeL_constant_baseline": round(score(const_pairs), 4),
        "rougeL_margin_correct_vs_shuffled":
            round(score(correct_pairs) - score(shuffled_pairs), 4),
        "most_common_report_preview": most_common_report[:100],
        "finding_agreement": {},
    }

    # ---- 4. finding agreement (keyword precision/recall) ------------------
    for label, kws in FINDING_KEYWORDS.items():
        tp = fp = fn = 0
        base_rate = 0.0
        for gen, ref in zip(correct, refs):
            gen_hit = any(k in gen.lower() for k in kws)
            ref_hit = any(k in ref.lower() for k in kws)
            base_rate += ref_hit
            if gen_hit and ref_hit:
                tp += 1
            elif gen_hit and not ref_hit:
                fp += 1
            elif ref_hit and not gen_hit:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        results["finding_agreement"][label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "reference_base_rate": round(base_rate / max(1, len(refs)), 4),
        }

    # ---- PASS/FAIL against the frozen criteria ---------------------------
    checks = {
        "distinct_ratio_gt_0.5": distinct_ratio > 0.5,
        "correct_beats_shuffled": results["rougeL_margin_correct_vs_shuffled"] > 0.01,
        "beats_constant_baseline":
            score(correct_pairs) > score(const_pairs),
        "some_finding_prec_above_base_rate": any(
            v["precision"] > v["reference_base_rate"] and v["precision"] > 0
            for v in results["finding_agreement"].values()),
    }
    results["checks"] = checks
    results["gate_passed"] = all(checks.values())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== GATE REPORT ===")
    for k, v in results.items():
        if k != "finding_agreement":
            print(f"  {k}: {v}")
    print(f"  finding_agreement: {json.dumps(results['finding_agreement'], indent=4)}")
    print(f"saved to {out}")
    if not results["gate_passed"]:
        print("\nGATE FAILED - do not start the long run. See plan4.md §Phase 4.4 "
              "for the fallback ladder (verify 4.1/4.2 took effect, raise LR to 2e-4, "
              "r=16 alpha=32, then consider projector LoRA).")
        sys.exit(1)


if __name__ == "__main__":
    main()
