"""
Grad-CAM Visualization Test
============================
Loads the trained model and generates Grad-CAM heatmaps for sample images
from the test set. Saves visualizations to gradcam_outputs/.

Usage:
    python gradcam_test.py                    # default: 10 samples
    python gradcam_test.py --num_samples 5    # custom count
    python gradcam_test.py --image path.png   # single image
"""

import argparse
import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.explainability.gradcam import (
    load_model_for_gradcam,
    preprocess_image,
    predict_with_confidence,
    generate_gradcam_for_all_classes,
)
from dataset import load_and_merge_data, build_label_encoder


OUTPUT_DIR = Path("gradcam_outputs")


def get_test_images(n=10):
    """Grab n sample images from the test set."""
    df = load_and_merge_data()
    label_names, label_vectors = build_label_encoder(df)

    from sklearn.model_selection import train_test_split
    unique_uids = df["uid"].unique()
    trainval_uids, test_uids = train_test_split(
        unique_uids, test_size=0.15, random_state=42
    )
    train_uids, _ = train_test_split(
        trainval_uids, test_size=0.214, random_state=42
    )
    test_mask = df["uid"].isin(test_uids)
    test_df = df[test_mask].reset_index(drop=True)

    # Pick n samples spread across the dataset
    indices = np.linspace(0, len(test_df) - 1, n, dtype=int)

    samples = []
    for idx in indices:
        row = test_df.iloc[idx]
        img_path = Path("data/images/preprocessed") / row["filename"]
        if img_path.exists():
            # Get ground truth labels
            gt_labels = [label_names[j]
                         for j in range(len(label_names))
                         if label_vectors[test_mask.values][idx, j] == 1.0]
            samples.append({
                "path": img_path,
                "uid": row["uid"],
                "gt_labels": gt_labels,
            })

    return samples, label_names


def save_overlay(overlay, save_path):
    """Save a numpy overlay array as a PNG."""
    img = (overlay * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def run_gradcam_single(model, image_path, label_names, thresholds, output_dir):
    """Generate and save Grad-CAM for a single image, all predicted classes."""
    results, rgb_np = generate_gradcam_for_all_classes(
        model, image_path, label_names, thresholds, min_prob=0.1
    )

    img_stem = Path(image_path).stem
    img_dir = output_dir / img_stem
    img_dir.mkdir(parents=True, exist_ok=True)

    # Save the original image
    save_overlay(rgb_np, img_dir / "original.png")

    # Save a summary text
    lines = [f"Image: {image_path.name}", ""]
    for r in results:
        flag = " [FLAGGED]" if r["flagged"] else ""
        lines.append(f"  {r['name']:25s}  prob={r['prob']:.3f}  thresh={r['threshold']:.2f}{flag}")
        save_overlay(r["overlay"], img_dir / f"{r['name']}.png")

    (img_dir / "predictions.txt").write_text("\n".join(lines))
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Grad-CAM on sample images")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, label_names, thresholds = load_model_for_gradcam(args.checkpoint)
    print(f"  Classes: {label_names}")

    if args.image:
        # Single image mode
        image_path = Path(args.image)
        print(f"\nRunning Grad-CAM on: {image_path}")
        results = run_gradcam_single(model, image_path, label_names, thresholds, OUTPUT_DIR)
        for r in results:
            flag = " [+]" if r["flagged"] else ""
            print(f"  {r['name']:25s}  {r['prob']:.1%}{flag}")
        print(f"\nOutputs saved to: {OUTPUT_DIR / image_path.stem}/")
    else:
        # Test set samples
        print(f"\nFetching {args.num_samples} test images...")
        samples, _ = get_test_images(args.num_samples)
        print(f"  Got {len(samples)} images\n")

        for i, sample in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] {sample['path'].name}")
            print(f"  GT labels: {sample['gt_labels'] or '(none)'}")

            results = run_gradcam_single(
                model, sample["path"], label_names, thresholds, OUTPUT_DIR
            )
            for r in results[:5]:  # top 5 predictions
                flag = " [+]" if r["flagged"] else ""
                print(f"  {r['name']:25s}  {r['prob']:.1%}{flag}")
            print()

        print(f"All outputs saved to: {OUTPUT_DIR}/")

    # Print directory listing
    print(f"\nGenerated files:")
    for p in sorted(OUTPUT_DIR.rglob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
