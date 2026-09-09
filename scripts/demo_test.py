"""Demo sanity‑check script for the end‑to‑end pipeline.

Running this script will:
  1. Load a sample chest X‑ray image from the ``data/images/preprocessed`` directory.
  2. Run the classifier and display the top predictions.
  3. Generate a Grad‑CAM overlay for the top prediction.
  4. Produce a generated report using the fine‑tuned Qwen‑2‑VL LoRA model.
  5. Run the hallucination detection logic and print any flags.

The script is intended for quick verification and can be used in CI.
"""
import os
import sys
from pathlib import Path
import json
import tempfile

# Ensure the ``src`` folder is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from classifier.predict import predict, load_classifier, preprocess_image
from explainability.gradcam import generate_gradcam
from inference.pipeline import run_report
from hallucination.check import extract_findings, compare_hallucination

def main():
    # Locate a sample image.
    images_dir = PROJECT_ROOT / "data" / "images" / "preprocessed"
    sample_imgs = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if not sample_imgs:
        print("No preprocessed images found – aborting demo test.")
        return
    img_path = sample_imgs[0]
    print(f"Using sample image: {img_path.name}\n")

    # ---------- Classifier ----------
    probs = predict(str(img_path))
    # Show top‑3 predictions.
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print("Top classifier predictions (probability):")
    for label, prob in top3:
        print(f"  {label:25s}: {prob:.3f}")
    predicted_labels = [label for label, p in probs.items() if p >= 0.5]

    # ---------- Grad‑CAM (for the highest‑probability label) ----------
    if predicted_labels:
        from classifier.predict import preprocess_image
        img_tensor = preprocess_image(img_path)
        classifier_model = load_classifier()
        label_to_idx = {lbl: idx for idx, lbl in enumerate(classifier_model.pathologies)}
        top_label = predicted_labels[0]
        class_idx = label_to_idx[top_label]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_cam:
            cam_path = generate_gradcam(classifier_model, img_tensor, class_idx, tmp_cam.name)
            print(f"\nGrad‑CAM written to: {cam_path}")
        # Cleanup Grad‑CAM file after display.
        os.remove(cam_path)
    else:
        print("No predictions above threshold – skipping Grad‑CAM.")

    # ---------- Report generation ----------
    report = run_report(str(img_path))
    print("\nGenerated report:\n")
    print(report)

    # ---------- Hallucination detection ----------
    vocab = list(probs.keys())
    mentioned = extract_findings(report, vocab)
    hallucinations = compare_hallucination(set(predicted_labels), mentioned)
    if hallucinations:
        print("\nHallucination flags:")
        for msg in hallucinations:
            print(f"  - {msg}")
    else:
        print("\nNo hallucinations detected.")

if __name__ == "__main__":
    main()
