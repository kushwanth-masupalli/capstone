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
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel


CHECKPOINT_DIR = Path("checkpoints/qwen_finetune_patient_split")

# Must EXACTLY match PROMPT_TEXT in src/generation/finetune.py: the model was
# LoRA-trained on this instruction, so inference must use the same one.
PROMPT_TEXT = "Write a chest X-ray report with Findings and Impression."


def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(
        CHECKPOINT_DIR,
        trust_remote_code=True
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
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
                    "text": PROMPT_TEXT
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

    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    # ---------------------------------------------------------
    # Decode — ONLY the newly generated tokens.
    #
    # generated_ids initially contains the full sequence (prompt +
    # generation). Decoding the whole thing leaks the chat-template
    # boilerplate ("system\nYou are a helpful assistant.\nuser\n...") into
    # the "report" text shown in the UI and fed to the hallucination
    # checker. Slice off the prompt tokens before decoding.
    # ---------------------------------------------------------

    generated_ids_trimmed = generated_ids[:, input_len:]

    generated = processor.decode(
        generated_ids_trimmed[0],
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
    # Run pipeline (reuse cached model if possible)
    # ---------------------------------------------------------

    generated = run_report(str(image_path))

    # ---------------------------------------------------------
    # Build result – minimal fields for the demo UI
    # ---------------------------------------------------------

    result = {
        "image_path": str(image_path.resolve()),
        "generated_report": generated,
        # The UI can fill the remaining fields if desired.
    }

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Result written to {out_path}")


def _load_cached():
    """Load processor and model once per process and cache them globally.

    The function returns a tuple (processor, model). Subsequent calls reuse the
    same objects, which speeds up repeated UI invocations.
    """
    global _cached_processor, _cached_model
    if _cached_processor is None or _cached_model is None:
        _cached_processor, _cached_model = load_model_and_processor()
    return _cached_processor, _cached_model

# Global cache placeholders
_cached_processor = None
_cached_model = None

def run_report(image_path: str) -> str:
    """Generate a report for ``image_path`` using the fine‑tuned LoRA model.

    This helper is used by the Streamlit UI to avoid the overhead of a subprocess
    call. It returns only the generated report string.
    """
    processor, model = _load_cached()
    return generate_report(processor, model, Path(image_path))


if __name__ == "__main__":
    main()