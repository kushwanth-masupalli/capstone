#!/usr/bin/env python3
"""Simple inference wrapper for the fine‑tuned report generation model.

The wrapper loads the LoRA‑adapted Qwen‑2‑VL‑2B model and its processor, runs the
model on a single image, and returns a JSON structure compatible with the later
demo UI.

CLI usage::

    python -m src.inference.pipeline \
        --image path/to/xray.png \
        --output out.json

The output JSON contains:
    {
        "image_path": "...",
        "predicted_findings": [],   # placeholder – classifier not implemented yet
        "generated_report": "...",
        "gradcam_path": null,       # placeholder – Grad‑CAM not yet implemented
        "hallucination_flags": []   # placeholder – to be added later
    }
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

CHECKPOINT_DIR = Path("checkpoints/qwen_finetune_patient_split")

def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()
    return processor, model

def generate_report(processor, model, image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    user_msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Write a chest X‑ray report with Findings and Impression."},
        ],
    }]
    prompt = processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = processor.decode(generated_ids[0], skip_special_tokens=True)
    return generated.strip()

def main():
    parser = argparse.ArgumentParser(description="Run the report generation model on a single image.")
    parser.add_argument("--image", required=True, help="Path to the X‑ray image")
    parser.add_argument("--output", required=True, help="Path to JSON file to write results")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    processor, model = load_model_and_processor()
    generated = generate_report(processor, model, image_path)

    result = {
        "image_path": str(image_path.resolve()),
        "predicted_findings": [],  # placeholder for future classifier output
        "generated_report": generated,
        "gradcam_path": None,       # placeholder – Grad‑CAM to be added later
        "hallucination_flags": [], # placeholder – detection module to be added later
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Result written to {out_path}")

if __name__ == "__main__":
    main()
