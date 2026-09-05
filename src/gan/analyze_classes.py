"""
Phase 1 — Choose GAN target classes
====================================
Prints per-class real-image counts in the TRAIN split only, so you can pick
the 2-3 rarest classes that still have enough real images to train a GAN
(rule of thumb: don't target anything under ~50-80 real training images).

Usage:
    python -m src.gan.analyze_classes
"""

import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset import get_splits


def main():
    df, label_names, label_vectors, train_mask, val_mask, test_mask = get_splits()

    train_vecs = label_vectors[train_mask]
    val_vecs = label_vectors[val_mask]
    test_vecs = label_vectors[test_mask]

    print("\n" + "=" * 70)
    print("PHASE 1 — Per-class real image counts (train split only)")
    print("=" * 70)
    print(f"{'Label':<32}{'Train':>8}{'Val':>8}{'Test':>8}  Candidate?")
    print("-" * 70)

    rows = []
    for i, name in enumerate(label_names):
        n_train = int(train_vecs[:, i].sum())
        n_val = int(val_vecs[:, i].sum())
        n_test = int(test_vecs[:, i].sum())
        rows.append((name, n_train, n_val, n_test))

    # Sort by train count ascending — rarest first
    rows.sort(key=lambda r: r[1])

    for name, n_train, n_val, n_test in rows:
        # Candidate = rare but above the ~50-80 real-image floor
        candidate = 50 <= n_train <= 350
        flag = "  << candidate" if candidate else ""
        print(f"{name:<32}{n_train:>8}{n_val:>8}{n_test:>8}{flag}")

    print("-" * 70)
    print("\nGuidance:")
    print("  - Candidate classes: rare (low train count) but >= ~50 real train images.")
    print("  - Pick 2-3 classes; these are the ones hurting per-class F1 most.")
    print("  - 'normal' is not a pathology target and is excluded by design.")
    print("  - The GAN will be trained ONLY on these train-split images;")


if __name__ == "__main__":
    main()