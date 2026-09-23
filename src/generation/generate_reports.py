#!/usr/bin/env python3
"""Generate reports for a CSV split with the fine-tuned MedGemma LoRA model.

Rewired per PLAN 4 (Instruction 7): the model class, processor, base-model id,
prompt text and checkpoint dir all come from ``medgemma_io.py``, so training
and inference can never drift apart.

Usage:
    python src/generation/generate_reports.py \
        --split data/report_splits/test.csv \
        --output results/medgemma_patient_split/pipeline_results.json \
        --limit 5

The TorchXRayVision classifier still runs per image to fill
``predicted_findings`` for the hallucination checker and Grad-CAM. Its output
is NOT put in the prompt (classifier grounding was removed in ``dd428b6``).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow `from classifier...`
from classifier.predict import load_classifier, preprocess_image  # noqa: E402

import torch  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from generation.medgemma_io import (  # noqa: E402
    PROMPT_TEXT,
    generate_report_for_image,
    load_image,
    load_model_and_processor,
)

# Probability above which a pathology is recorded as a candidate finding in the
# JSON (for the hallucination checker / demo). Not injected into the prompt.
FINDING_THRESHOLD = 0.5
MAX_FINDINGS = 5


def get_predicted_findings(classifier, image_path: Path):
    """Top pathologies above FINDING_THRESHOLD from the TorchXRayVision classifier."""
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        logits = classifier(img_tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    pairs = list(zip(classifier.pathologies, probs))
    pairs = [(label, float(p)) for label, p in pairs if label and p >= FINDING_THRESHOLD]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [label for label, _ in pairs[:MAX_FINDINGS]]


def generate_one(processor, model, image_path: Path):
    image = load_image(image_path)
    return generate_report_for_image(processor, model, image, PROMPT_TEXT)


def main():
    parser = argparse.ArgumentParser(description="Generate reports for a CSV split (MedGemma).")
    parser.add_argument("--split", required=True,
                        help="CSV with test split (sample_id,uid,image_path,report_text)")
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (smoke test).")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Skip the TorchXRayVision classifier entirely.")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.split)
    required = {"sample_id", "uid", "image_path", "report_text"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    if args.limit:
        df = df.head(args.limit)

    processor, model = load_model_and_processor()
    classifier = None if args.no_classifier else load_classifier()

    results = []
    for _, row in df.iterrows():
        img_path = Path(str(row["image_path"]).replace("\\", "/"))
        if not img_path.is_file():
            # Fall back to the VLM column / raw images dir (Phase 4.2 CSVs).
            for cand in [Path(str(row.get("vlm_image_path", "")).replace("\\", "/")),
                         Path("data/images") / img_path.name]:
                if str(cand) and cand.is_file():
                    img_path = cand
                    break
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        generated = generate_one(processor, model, img_path)
        findings = get_predicted_findings(classifier, img_path) if classifier else []
        results.append({
            "sample_id": row["sample_id"],
            "uid": row["uid"],
            "image_path": str(img_path),
            "reference_report": row["report_text"],
            "generated_report": generated,
            "predicted_findings": findings,
        })
        print(f"Processed {img_path.name}  |  findings: {findings}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"All done - results saved to {out_path}")


if __name__ == "__main__":
    main()
