"""
Phase 3 — Generate + filter pseudo-labels with the teacher model
================================================================
Runs an IU-trained teacher classifier over the unlabeled pool from Phase 1
(data/pseudo_pool/), and keeps a pseudo-label for a class ONLY when the
teacher's probability for it clears a HIGH confidence threshold (default
0.85 — the whole point of self-training is that wrong pseudo-labels teach
the model to be confidently wrong, so bias hard toward precision).

Outputs:
    data/pseudo_labels.csv            kept images + one 0/1 column per label
                                      (+ a _prob column per label for audit)
    data/pseudo_probs.npz             raw teacher probabilities for every pool
                                      image (re-run filters without re-inferring)
    data/pseudo_pool/_review/<Label>.png   contact sheet of ~20 kept images per
                                      class — VISUALLY SPOT-CHECK these before
                                      training (Phase 3 checkpoint)

Images where NO class clears its threshold are discarded (not force-labeled).

Usage:
    python -m src.pseudo.generate_pseudo_labels --teacher checkpoints/base_best_model.pth

    # Lower the bar only for specific rare classes (never uniformly to 0.5):
    python -m src.pseudo.generate_pseudo_labels \
        --class_thresholds '{"Cardiomegaly": 0.80, "Pulmonary Hypoinflation": 0.80}'
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset import PATHOLOGY_LABELS, get_eval_transform, XRVPreprocessor
from train import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POOL_DIR = Path("data/pseudo_pool")
OUT_CSV = Path("data/pseudo_labels.csv")
OUT_NPZ = Path("data/pseudo_probs.npz")
REVIEW_DIR = Path("data/pseudo_pool/_review")


def load_teacher(checkpoint_path):
    """Load checkpoint + rebuild the model. Returns (model, label_names, backbone)."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    label_names = ckpt.get("label_names")
    if label_names is None:
        sys.exit(f"ERROR: {checkpoint_path} has no 'label_names' — is this a train.py checkpoint?")
    if list(label_names) != PATHOLOGY_LABELS:
        sys.exit(
            f"ERROR: teacher taxonomy mismatch.\n"
            f"  checkpoint: {list(label_names)}\n"
            f"  current:    {PATHOLOGY_LABELS}\n"
            f"The teacher must be trained on the CURRENT 7-pathology IU taxonomy "
            f"(e.g. checkpoints/base_best_model.pth)."
        )
    backbone = ckpt.get("backbone", "imagenet")
    model = create_model(
        num_labels=len(label_names),
        backbone=backbone,
        xrv_checkpoint=ckpt.get("xrv_checkpoint", "densenet121-res224-chex"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Teacher: {Path(checkpoint_path).name}  backbone={backbone}  "
          f"labels={len(label_names)}")
    return model, label_names, backbone


def infer_pool(model, pool_files, transform, batch_size):
    """Teacher probabilities for every pool image -> (names, probs np.ndarray)."""
    probs_list, names = [], []
    model = model.to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(pool_files), batch_size):
            chunk = pool_files[i : i + batch_size]
            tensors = [transform(Image.open(p).convert("RGB")) for p in chunk]
            batch = torch.stack(tensors).to(DEVICE)
            logits = model(batch)
            probs_list.append(torch.sigmoid(logits[:, : len(PATHOLOGY_LABELS)]).cpu().numpy())
            names.extend(p.name for p in chunk)
            print(f"  inferred {min(i + batch_size, len(pool_files))}/{len(pool_files)}")
    return names, np.concatenate(probs_list, axis=0)


def save_contact_sheets(pool_dir, manifest_df, label_names, out_dir, per_class=20, seed=42):
    """Tile up to `per_class` kept pool images per label for visual review."""
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for j, name in enumerate(label_names):
        kept = manifest_df.loc[manifest_df[name] == 1]
        if kept.empty:
            continue
        kept = kept.sample(n=min(per_class, len(kept)), random_state=rng)
        thumbs = []
        for _, row in kept.iterrows():
            img = Image.open(pool_dir / row["filename"]).convert("L")
            thumbs.append(img)
        cols = 5
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("L", (cols * 224, rows * 224 + 20), 255)
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 2), f"{name}  (pseudo-labeled)", fill=0)
        for k, img in enumerate(thumbs):
            r, c = divmod(k, cols)
            sheet.paste(img, (c * 224, r * 224 + 20))
        out = out_dir / f"{name}.png"
        sheet.save(out)
        print(f"  contact sheet: {out} ({len(thumbs)} images)")


def main():
    parser = argparse.ArgumentParser(description="Pseudo-label the NIH pool with the teacher")
    parser.add_argument("--teacher", type=str, default="checkpoints/base_best_model.pth",
                        help="IU-trained teacher checkpoint (train.py format)")
    parser.add_argument("--pool_dir", type=str, default=str(POOL_DIR))
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Global confidence threshold for keeping a class (0.85-0.90)")
    parser.add_argument("--class_thresholds", type=str, default="",
                        help="Per-class overrides as JSON, e.g. "
                             '{"Cardiomegaly": 0.80, "Pulmonary Hypoinflation": 0.80}')
    parser.add_argument("--out_csv", type=str, default=str(OUT_CSV))
    parser.add_argument("--prob_out", type=str, default=str(OUT_NPZ),
                        help="Path for raw probs npz ('' = don't save)")
    parser.add_argument("--review_dir", type=str, default=str(REVIEW_DIR))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=0,
                        help="Cap pool size (smoke tests only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pool_dir = Path(args.pool_dir)
    if not pool_dir.is_dir():
        print(f"ERROR: pool folder not found: {pool_dir}")
        print("Run Phase 1 first:  python -m src.pseudo.prepare_pool --source <nih raw folder>")
        sys.exit(1)
    pool_files = sorted(pool_dir.glob("*.png"))
    if not pool_files:
        print(f"ERROR: no PNGs in {pool_dir} - run Phase 1 first.")
        sys.exit(1)
    if args.max_images:
        pool_files = pool_files[: args.max_images]
    print(f"Pool: {len(pool_files)} images from {pool_dir}")

    # Per-class thresholds = global threshold + optional JSON overrides
    thresholds = np.full(len(PATHOLOGY_LABELS), args.threshold, dtype=np.float64)
    if args.class_thresholds:
        overrides = json.loads(args.class_thresholds)
        for name, t in overrides.items():
            if name not in PATHOLOGY_LABELS:
                print(f"WARNING: '{name}' not in taxonomy, ignoring override")
                continue
            thresholds[PATHOLOGY_LABELS.index(name)] = t

    # ── Teacher inference ──
    model, label_names, backbone = load_teacher(Path(args.teacher))
    transform = XRVPreprocessor(train=False) if backbone == "xrv" else get_eval_transform()
    names, probs = infer_pool(model, pool_files, transform, args.batch_size)

    if args.prob_out:
        np.savez(Path(args.prob_out), names=np.array(names), probs=probs,
                 label_names=np.array(label_names), thresholds=thresholds)
        print(f"Raw probs saved: {args.prob_out}")

    # ── Confidence filtering ──
    kept_flags = probs >= thresholds[np.newaxis, :]
    any_kept = kept_flags.any(axis=1)
    print(f"\n--- Confidence filter (threshold per class) ---")
    for i, name in enumerate(label_names):
        print(f"  {name:28s} thresh={thresholds[i]:.2f}  "
              f"kept {int(kept_flags[:, i].sum()):5d}/{len(pool_files)}")
    print(f"  {'(discarded: no class above threshold)':28s}        "
          f"{int((~any_kept).sum()):5d}/{len(pool_files)}")
    print(f"  TOTAL kept images: {int(any_kept.sum())}")

    # ── Write filtered manifest (train-set input for Phase 4) ──
    df = pd.DataFrame({"filename": names})
    for i, name in enumerate(label_names):
        df[name] = kept_flags[:, i].astype(int)
        df[f"{name}_prob"] = np.round(probs[:, i], 4)
    df = df[any_kept].reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nPseudo-label manifest: {out_csv}  ({len(df)} images, "
          f"{int(df[label_names].values.sum())} positive labels)")

    save_contact_sheets(pool_dir, df, label_names, args.review_dir, seed=args.seed)
    print("\nSPOT-CHECK the contact sheets in " + str(Path(args.review_dir)) +
          " before Phase 4 - if a class looks like garbage, lower its "
          "--class_thresholds bar or drop it from training.")

    print("\nNext step:")
    print("  python train.py --backbone xrv --epochs 80 --balanced_sampling \\")
    print("      --focal_loss --save_prefix pseudo_ --pseudo_csv data/pseudo_labels.csv")


if __name__ == "__main__":
    main()
