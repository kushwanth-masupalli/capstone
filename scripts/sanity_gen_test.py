#!/usr/bin/env python3
"""Sanity test: compare generation configs on a few test images (MedGemma).

Verifies that bad outputs (markdown boilerplate, loops) are caused by
no_repeat_ngram_size / repetition_penalty, not the model itself.

Rewired per PLAN 4 (Instruction 7) to load the MedGemma base model and the
``checkpoints/medgemma_finetune`` adapter via ``medgemma_io.py``.
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "generation"))
from generation.medgemma_io import (  # noqa: E402
    ADAPTER_DIR,
    PROMPT_TEXT,
    load_model_and_processor,
)

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

    processor, model = load_model_and_processor(ADAPTER_DIR)

    for label, kwargs in CONFIGS.items():
        print("\n" + "=" * 70)
        print(f"CONFIG: {label}")
        print("=" * 70)
        for _, row in df.iterrows():
            from PIL import Image
            image = Image.open(Path(str(row["image_path"]).replace("\\", "/"))).convert("RGB")
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEXT},
                ]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # add_special_tokens=False: the template already emitted <bos> (PLAN 4, C3).
            inputs = processor(text=[prompt], images=[image], add_special_tokens=False,
                               return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
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
