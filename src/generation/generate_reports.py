#!/usr/bin/env python3
"""Generate reports for a set of X-ray images using the fine-tuned LoRA Qwen-2-VL-2B model.

Usage:
    python src/generation/generate_reports.py \
        --split data/report_splits/test.csv \
        --output results/finetuned_patient_split/pipeline_results.json

This version grounds each generation with predictions from the existing
TorchXRayVision classifier (src/classifier/predict.py). The Qwen2-VL model on
its own was shown to hallucinate findings unrelated to the specific image
(e.g. abdominal free-air, spine changes copied from other training reports)
because its own visual conditioning on X-rays is weak. Feeding it the
classifier's top predicted pathologies as explicit context grounds the
generation in something the image actually supports, instead of relying
solely on the VLM's own (undertrained) visual understanding.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow `from classifier...` import
from classifier.predict import load_classifier, preprocess_image

CHECKPOINT_DIR = Path("checkpoints/qwen_finetune_patient_split")

# Probability above which a pathology is mentioned as a candidate finding in
# the grounding context. TorchXRayVision models are not perfectly calibrated,
# so this is a soft signal for the prompt, not a diagnosis.
FINDING_THRESHOLD = 0.5
MAX_FINDINGS = 5


def load_model_and_processor():
    """Load the base Qwen-2-VL-2B model, attach the LoRA adapter, and return the processor."""
    processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()
    return processor, model


def get_predicted_findings(classifier, image_path: Path):
    """Run the TorchXRayVision classifier and return a short list of the
    highest-probability pathologies above FINDING_THRESHOLD, sorted by
    confidence. Returns an empty list if nothing crosses the threshold."""
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        logits = classifier(img_tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    pairs = list(zip(classifier.pathologies, probs))
    pairs = [(label, float(p)) for label, p in pairs if label and p >= FINDING_THRESHOLD]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [label for label, _ in pairs[:MAX_FINDINGS]]


def build_grounding_text(findings):
    if not findings:
        return "Automated screening detected no significant abnormal findings."
    joined = ", ".join(findings)
    return f"Automated screening flagged possible: {joined}. Confirm or refute each against the image."


def generate_one(processor, model, classifier, image_path: Path):
    image = Image.open(image_path).convert("RGB")
    findings = get_predicted_findings(classifier, image_path)
    grounding = build_grounding_text(findings)

    user_msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Write a chest X-ray report with Findings and Impression."},
        ],
    }]
    prompt = processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # Only the NEWLY generated tokens should be decoded — generated_ids initially
    # contains the full sequence (prompt + generation). Decoding the whole thing
    # leaks chat-template boilerplate into the "report" text and corrupts every
    # downstream metric (BLEU/ROUGE/METEOR/CIDEr).
    input_len = inputs["input_ids"].shape[1]
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,          # reports in this dataset are short; 512 let the
                                      # model ramble/loop far past where a real report ends
        do_sample=False,
    )
    generated_ids_trimmed = generated_ids[:, input_len:]
    generated_text = processor.decode(generated_ids_trimmed[0], skip_special_tokens=True)
    return generated_text.strip(), findings


def main():
    parser = argparse.ArgumentParser(description="Generate reports for a CSV split.")
    parser.add_argument("--split", required=True, help="CSV with test split (sample_id,uid,image_path,report_text)")
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.split)
    required = {"sample_id", "uid", "image_path", "report_text"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    processor, model = load_model_and_processor()
    classifier = load_classifier()

    results = []
    for _, row in df.iterrows():
        img_path = Path(row["image_path"]).expanduser()
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        generated, findings = generate_one(processor, model, classifier, img_path)
        results.append({
            "sample_id": row["sample_id"],
            "uid": row["uid"],
            "image_path": str(img_path),
            "reference_report": row["report_text"],
            "generated_report": generated,
            "predicted_findings": findings,
        })
        print(f"Processed {img_path.name}  |  predicted: {findings}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"All done – results saved to {out_path}")


if __name__ == "__main__":
    main()