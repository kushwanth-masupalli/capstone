#!/usr/bin/env python3
"""Generate reports for a set of X‑ray images using the fine‑tuned LoRA Qwen‑2‑VL‑2B model.

Usage:
    python src/generation/generate_reports.py \
        --split data/report_splits/test.csv \
        --output results/finetuned_patient_split/pipeline_results.json

The script:
  1. Loads the LoRA‑adapted model and processor from the checkpoint directory
     `checkpoints/qwen_finetune_patient_split` (produced by `finetune.py`).
  2. Reads the CSV `split` which must contain the columns:
     `sample_id, uid, image_path, report_text` (same format as created by
     `create_finetune_csv.py`).
  3. For each row, runs the model on the image and captures the generated
     report.
  4. Writes a JSON list where each element is a dict:
     {
       "sample_id": <sample_id>,
       "uid": <uid>,
       "image_path": <image_path>,
       "reference_report": <report_text>,
       "generated_report": <generated_text>
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
    """Load the base Qwen‑2‑VL‑2B model, attach the LoRA adapter, and return the processor."""
    processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Load LoRA weights saved by finetune.py.
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()
    return processor, model

def generate_one(processor, model, image_path: Path) -> str:
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
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text.strip()

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
    results = []
    for _, row in df.iterrows():
        img_path = Path(row["image_path"]).expanduser()
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        generated = generate_one(processor, model, img_path)
        results.append({
            "sample_id": row["sample_id"],
            "uid": row["uid"],
            "image_path": str(img_path),
            "reference_report": row["report_text"],
            "generated_report": generated,
        })
        print(f"Processed {img_path.name}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"All done – results saved to {out_path}")

if __name__ == "__main__":
    main()
