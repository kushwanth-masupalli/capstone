#!/usr/bin/env python3
"""Simple inference wrapper for the fine-tuned MedGemma report generator.

Rewired per PLAN 4 (Instruction 7): model class, prompt text, processor and
checkpoint dir all come from ``medgemma_io.py``. The LoRA adapter is loaded
from ``checkpoints/medgemma_finetune``; with no adapter present the zero-shot
base model is used (a warning is printed).
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # put `src` on the path
from generation.medgemma_io import (  # noqa: E402
    ADAPTER_DIR,
    generate_report_for_image,
    load_image,
    load_model_and_processor,
)

_cached_processor = None
_cached_model = None


def load_model_and_processor_cached():
    """Load processor and model once per process (used by the Streamlit UI)."""
    global _cached_processor, _cached_model
    if _cached_processor is None or _cached_model is None:
        _cached_processor, _cached_model = load_model_and_processor(ADAPTER_DIR)
    return _cached_processor, _cached_model


def generate_report(processor, model, image_path: Path) -> str:
    image = load_image(image_path)
    return generate_report_for_image(processor, model, image)


def run_report(image_path: str) -> str:
    """Generate a report for ``image_path`` (helper used by the Streamlit UI)."""
    processor, model = load_model_and_processor_cached()
    return generate_report(processor, model, Path(image_path))


def main():
    parser = argparse.ArgumentParser(
        description="Run the MedGemma report generator on a single image."
    )
    parser.add_argument("--image", required=True, help="Path to the X-ray image")
    parser.add_argument("--output", required=True, help="Path to JSON file to write results")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    generated = run_report(str(image_path))

    result = {
        "image_path": str(image_path.resolve()),
        "generated_report": generated,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Result written to {out_path}")


if __name__ == "__main__":
    main()
