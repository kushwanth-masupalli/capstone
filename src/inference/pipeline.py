#!/usr/bin/env python3
"""Simple inference wrapper for the fine-tuned report generation model.

The wrapper loads the LoRA-adapted Qwen-2-VL-2B model and its processor,
runs the model on a single image, and returns a JSON structure compatible
with the later demo UI.
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
    processor = AutoProcessor.from_pretrained(
        CHECKPOINT_DIR,
        trust_remote_code=True
    )

    base_model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        CHECKPOINT_DIR
    )

    model.eval()

    return processor, model


def generate_report(processor, model, image_path: Path) -> str:

    # ---------------------------------------------------------
    # Load and resize image
    # ---------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # Keep the image small enough for a 6 GB GPU.
    # This significantly reduces the number of visual tokens.
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)

    # ---------------------------------------------------------
    # Build chat message
    # ---------------------------------------------------------

    user_msg = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": "Write a chest X-ray report with Findings and Impression."
                },
            ],
        }
    ]

    # ---------------------------------------------------------
    # Create prompt
    # ---------------------------------------------------------

    prompt = processor.apply_chat_template(
        user_msg,
        tokenize=False,
        add_generation_prompt=True
    )

    # ---------------------------------------------------------
    # Process image + text
    # ---------------------------------------------------------

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    # Move tensors to GPU
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    # ---------------------------------------------------------
    # Generate report
    # ---------------------------------------------------------

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    # ---------------------------------------------------------
    # Decode
    # ---------------------------------------------------------

    generated = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return generated.strip()


def main():

    parser = argparse.ArgumentParser(
        description="Run the report generation model on a single image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the X-ray image"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to JSON file to write results"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate image
    # ---------------------------------------------------------

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    processor, model = load_model_and_processor()

    # ---------------------------------------------------------
    # Generate
    # ---------------------------------------------------------

    generated = generate_report(
        processor,
        model,
        image_path
    )

    # ---------------------------------------------------------
    # Build result
    # ---------------------------------------------------------

    result = {
        "image_path": str(image_path.resolve()),
        "predicted_findings": [],
        "generated_report": generated,
        "gradcam_path": None,
        "hallucination_flags": [],
    }

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------

    out_path = Path(args.output)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"Result written to {out_path}")


if __name__ == "__main__":
    main()