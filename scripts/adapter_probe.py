#!/usr/bin/env python3
"""Probe: is the LoRA adapter actually influencing generation? (MedGemma)

1. Prints whether target modules really contain active LoRA layers.
2. Generates from the bare base model vs the adapter-loaded model on the
   same image and compares the texts.

Rewired per PLAN 4 (Instruction 7) to load the MedGemma base model and the
``checkpoints/medgemma_finetune`` adapter via ``medgemma_io.py``.
"""

from pathlib import Path

import pandas as pd
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "generation"))
from generation.medgemma_io import (  # noqa: E402
    ADAPTER_DIR,
    PROMPT_TEXT,
    load_model_and_processor,
)

N_IMAGES = 2


def make_inputs(processor, image):
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_TEXT},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # add_special_tokens=False: the template already emitted <bos> (PLAN 4, C3).
    return processor(text=[prompt], images=[image], add_special_tokens=False,
                     return_tensors="pt")


def gen_text(processor, model, inputs, n=80):
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False)
    return processor.decode(out[0, input_len:], skip_special_tokens=True).strip()


def main() -> None:
    print("### Loading BASE model (no adapter) ...")
    processor, base = load_model_and_processor(adapter_dir=None)

    df = pd.read_csv("data/report_splits/test.csv").head(N_IMAGES)
    images = []
    for p in df["image_path"]:
        from PIL import Image
        images.append(Image.open(Path(str(p).replace("\\", "/"))).convert("RGB"))
    batches = [make_inputs(processor, img) for img in images]

    base_outputs = [gen_text(processor, base, b) for b in batches]

    print(f"### Attaching LoRA adapter from {ADAPTER_DIR} ...")
    model = load_model_and_processor(ADAPTER_DIR)[1]

    # Inspect a few target modules: do they have active lora layers?
    def find_module(root, suffix):
        for name, mod in root.named_modules():
            if name.endswith(suffix):
                return name, mod
        return None, None

    probes = [
        ("layers[0].self_attn.q_proj", find_module(model, "layers.0.self_attn.q_proj")[1]),
        ("layers[0].mlp.down_proj", find_module(model, "layers.0.mlp.down_proj")[1]),
    ]
    for name, module in probes:
        has_lora = module is not None and hasattr(module, "lora_A") \
            and getattr(module.lora_A, "default", None) is not None
        scaling = module.scaling.get("default", "N/A") if has_lora else "N/A"
        print(f"  {name}: class={type(module).__name__}, lora_attached={has_lora}, scaling={scaling}")

    adapter_outputs = [gen_text(processor, model, b) for b in batches]

    for i, row in df.iterrows():
        j = i  # df index aligns since we took head(N_IMAGES)
        print("\n" + "=" * 70)
        print(f"--- {row['sample_id']} ---")
        print(f"REF    : {row['report_text'][:140]}")
        print(f"BASE   : {base_outputs[j][:200]}")
        print(f"ADAPTER: {adapter_outputs[j][:200]}")
        print(f"IDENTICAL: {base_outputs[j] == adapter_outputs[j]}")


if __name__ == "__main__":
    main()
