#!/usr/bin/env python3
"""Sanity test: compare generation configs on a few test images.

Verifies that the bad outputs in results/finetuned_patient_split/
pipeline_results.json (markdown boilerplate) are caused by
no_repeat_ngram_size / repetition_penalty, not the model itself.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

CHECKPOINT_DIR = Path("checkpoints/qwen_finetune_patient_split")
N_IMAGES = 4

CONFIGS = {
    "current (rep=1.3, ngram=3)": dict(
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    ),
    "old sep-9 (plain greedy 512)": dict(
        max_new_tokens=512,
        do_sample=False,
    ),
    "fixed (greedy 200, no penalties)": dict(
        max_new_tokens=200,
        do_sample=False,
    ),
}


def main() -> None:
    df = pd.read_csv("data/report_splits/test.csv").head(N_IMAGES)

    processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()

    for label, kwargs in CONFIGS.items():
        print("\n" + "=" * 70)
        print(f"CONFIG: {label}")
        print("=" * 70)
        for _, row in df.iterrows():
            image = Image.open(row["image_path"]).convert("RGB")
            user_msg = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Write a chest X-ray report with Findings and Impression."},
                ],
            }]
            prompt = processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                out = model.generate(**inputs, **kwargs)
            text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
            print(f"\n--- {row['sample_id']} ---")
            print(f"REF: {row['report_text'][:160]}")
            print(f"GEN: {text[:300].replace(chr(10), ' | ')}")

    print("\nDONE", file=sys.stderr)


if __name__ == "__main__":
    main()
