#!/usr/bin/env python3
"""Probe: is the LoRA adapter actually influencing generation?

1. Prints whether target modules really contain active LoRA layers.
2. Generates from the bare base model vs the adapter-loaded model on the
   same image and compares the texts.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

CHECKPOINT_DIR = Path("checkpoints/qwen_finetune_patient_split")


def make_prompt_and_inputs(processor, image):
    user_msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Write a chest X-ray report with Findings and Impression."},
        ],
    }]
    prompt = processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    return processor(text=prompt, images=image, return_tensors="pt")


def gen(model, inputs, n=80):
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False)
    return model.processor_tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip() \
        if hasattr(model, "processor_tokenizer") else None


def main() -> None:
    processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)

    df = pd.read_csv("data/report_splits/test.csv").head(2)
    images = [Image.open(p).convert("RGB") for p in df["image_path"]]
    batches = [make_prompt_and_inputs(processor, img) for img in images]

    print("### Loading BASE model (no adapter) ...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", device_map={"": 0},
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    base.eval()

    base_outputs = []
    for b in batches:
        b = {k: v.to(base.device) if hasattr(v, "to") else v for k, v in b.items()}
        input_len = b["input_ids"].shape[1]
        with torch.inference_mode():
            out = base.generate(**b, max_new_tokens=80, do_sample=False)
        base_outputs.append(processor.decode(out[0, input_len:], skip_special_tokens=True).strip())

    print("### Attaching LoRA adapter ...")
    model = PeftModel.from_pretrained(base, CHECKPOINT_DIR)
    model.eval()

    # Inspect a few target modules: do they have active lora layers?
    inner = model.base_model.model.model.language_model
    probes = [
        ("layers[0].self_attn.q_proj", inner.layers[0].self_attn.q_proj),
        ("layers[0].mlp.down_proj", inner.layers[0].mlp.down_proj),
    ]
    for name, module in probes:
        has_lora = hasattr(module, "lora_A") and getattr(module.lora_A, "default", None) is not None
        scaling = module.scaling.get("default", "N/A") if has_lora else "N/A"
        print(f"  {name}: class={type(module).__name__}, lora_attached={has_lora}, scaling={scaling}")

    adapter_outputs = []
    for b in batches:
        b = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in b.items()}
        input_len = b["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**b, max_new_tokens=80, do_sample=False)
        adapter_outputs.append(processor.decode(out[0, input_len:], skip_special_tokens=True).strip())

    for i, row in df.iterrows():
        j = i  # df index aligns since we took head(2)
        print("\n" + "=" * 70)
        print(f"--- {row['sample_id']} ---")
        print(f"REF    : {row['report_text'][:140]}")
        print(f"BASE   : {base_outputs[j][:200]}")
        print(f"ADAPTER: {adapter_outputs[j][:200]}")
        print(f"IDENTICAL: {base_outputs[j] == adapter_outputs[j]}")


if __name__ == "__main__":
    main()
