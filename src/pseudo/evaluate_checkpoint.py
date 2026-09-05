"""
Phase 2 / 5 — Evaluate any train.py checkpoint on the REAL IU split
===================================================================
Computes the exact numbers the pseudo-labeling go/no-go bar is judged on,
on the untouched, real-only IU val/test split (Phase 0 rule):

    - test AUC (pathology labels)
    - test F1 (micro) at the checkpoint's tuned thresholds
    - plus macro F1, exact-match image accuracy and per-class F1

Use it to record the TEACHER's reference numbers (Phase 2) and later to
judge the pseudo-label-trained model (Phase 5):

    # Teacher reference (current best real-only model):
    python -m src.pseudo.evaluate_checkpoint --checkpoint checkpoints/base_best_model.pth

    # Expanded-data model, after Phase 4:
    python -m src.pseudo.evaluate_checkpoint --checkpoint checkpoints/pseudo_best_model.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset import (IndianaChestXRayDataset, get_eval_transform, get_splits,
                     XRVPreprocessor)
from train import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a train.py checkpoint on the real IU split")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/base_best_model.pth")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "both"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    label_names = ckpt["label_names"]
    backbone = ckpt.get("backbone", "imagenet")
    n_pathology = ckpt.get("n_pathology", len(label_names))
    thresholds = np.asarray(ckpt.get("thresholds", np.full(len(label_names), 0.5)))
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  backbone={backbone}  labels={len(label_names)}  n_pathology={n_pathology}")

    model = create_model(
        num_labels=len(label_names),
        backbone=backbone,
        xrv_checkpoint=ckpt.get("xrv_checkpoint", "densenet121-res224-chex"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    df, _, label_vectors, train_mask, val_mask, test_mask = get_splits()
    transform = XRVPreprocessor(train=False) if backbone == "xrv" else get_eval_transform()

    def run_split(name, mask):
        ds = IndianaChestXRayDataset(df[mask], label_vectors[mask], label_names,
                                     transform=transform, train=False, backbone=backbone)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        probs_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images.to(DEVICE))
                probs_list.append(torch.sigmoid(logits[:, :n_pathology]).cpu().numpy())
                labels_list.append(labels[:, :n_pathology].numpy())
        probs = np.concatenate(probs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        pnames = list(label_names[:n_pathology])

        preds = (probs > thresholds[:n_pathology][np.newaxis, :]).astype(np.float32)

        print(f"\n{'=' * 66}\n{name.upper()}  (pathology labels, real IU images only)\n{'=' * 66}")
        try:
            auc = roc_auc_score(labels, probs, average="micro")
        except ValueError:
            auc = float("nan")
        print(f"  Test AUC (micro):       {auc:.4f}")
        print(f"  Test F1  (micro):       {f1_score(labels, preds, average='micro', zero_division=0):.4f}")
        print(f"  Test F1  (macro):       {f1_score(labels, preds, average='macro', zero_division=0):.4f}")
        print(f"  Precision (micro):      {precision_score(labels, preds, average='micro', zero_division=0):.4f}")
        print(f"  Recall (micro):         {recall_score(labels, preds, average='micro', zero_division=0):.4f}")
        exact = accuracy_score(labels, preds)
        print(f"  Exact-match accuracy:   {exact:.4f}  (whole label vector must match)")

        print(f"\n  {'Label':<30}{'F1':>7}{'Prec':>7}{'Rec':>7}{'AUC':>7}{'Thr':>6}{'Sup':>6}")
        for i, nm in enumerate(pnames):
            f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
            pr = precision_score(labels[:, i], preds[:, i], zero_division=0)
            rc = recall_score(labels[:, i], preds[:, i], zero_division=0)
            try:
                a = roc_auc_score(labels[:, i], probs[:, i])
            except ValueError:
                a = float("nan")
            print(f"  {nm:<30}{f1:>7.3f}{pr:>7.3f}{rc:>7.3f}{a:>7.3f}"
                  f"{thresholds[i]:>6.2f}{int(labels[:, i].sum()):>6d}")
        return auc

    if args.split in ("val", "both"):
        run_split("VAL", val_mask)
    if args.split in ("test", "both"):
        run_split("TEST", test_mask)


if __name__ == "__main__":
    main()
