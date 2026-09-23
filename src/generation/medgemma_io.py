#!/usr/bin/env python3
"""Shared IO for the fine-tuned MedGemma report generator (PLAN 4, Instruction 7).

Every entry point that touches the model — ``finetune.py``,
``generate_reports.py``, ``src/inference/pipeline.py``, ``demo/app.py`` and the
``scripts/*`` sanity tools — must import the model id, adapter path, prompt
text, quantisation config and input-building helpers from THIS module, so the
train/inference prompt-drift bug that ``dd428b6`` patched by hand cannot recur.

Notes
-----
* ``add_special_tokens=False`` is used everywhere text that already went
  through ``apply_chat_template`` is tokenised — the template already emits
  ``<bos>``, and letting the tokenizer prepend a second one is off-distribution
  for the base model (PLAN 4, C3).
* The assistant turn ends with ``<end_of_turn>``, not ``<eos>`` (PLAN 4, C4).
* ``dtype=`` instead of the deprecated ``torch_dtype=`` (PLAN 4, C7).
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Single source of truth
# ---------------------------------------------------------------------------
MODEL_NAME = "google/medgemma-1.5-4b-it"          # gated HuggingFace repo
BASE_MODEL_DIR = Path("checkpoints/medgemma_finetune")
ADAPTER_DIR = BASE_MODEL_DIR                       # alias for readability

# Must stay identical between training targets and inference (one definition).
PROMPT_TEXT = "Write a chest X-ray report with Findings and Impression."
END_OF_TURN = "<end_of_turn>\n"

# Shared generation settings (greedy, short reports).
GENERATE_KWARGS = {
    "max_new_tokens": 200,   # IU-Xray reports are short; longer just loops
    "do_sample": False,
}

_DTYPE = torch.bfloat16


def build_quant_config(skip_vision_quant: bool = True) -> BitsAndBytesConfig:
    """4-bit NF4 config shared by training and inference.

    ``skip_vision_quant=True`` keeps the SigLIP vision tower and the projector
    in bf16 so their features are not degraded and no dequantisation runs on
    the hot path (PLAN 4, S3 / Instruction 11). Costs ~0.5-0.7 GB extra VRAM;
    pass ``False`` if a 6 GB card OOMs.
    """
    skip = ["vision_tower", "multi_modal_projector", "lm_head"] if skip_vision_quant else None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_DTYPE,
        llm_int8_skip_modules=skip,
    )


def load_model_and_processor(
    adapter_dir: Path | str | None = ADAPTER_DIR,
    quantize_4bit: bool = True,
    skip_vision_quant: bool = True,
):
    """Load the MedGemma base model and attach the LoRA adapter if present.

    Returns ``(processor, model)`` with the model in eval mode. Falls back to
    the zero-shot base model (with a warning) when ``adapter_dir`` does not
    contain an ``adapter_config.json``.
    """
    adapter_dir = Path(adapter_dir) if adapter_dir is not None else None
    has_adapter = bool(adapter_dir and (adapter_dir / "adapter_config.json").exists())
    if adapter_dir is not None and not has_adapter:
        print(f"WARNING: no adapter found at {adapter_dir} - using zero-shot {MODEL_NAME}")

    processor_src = adapter_dir if has_adapter and (Path(adapter_dir) / "preprocessor_config.json").exists() else MODEL_NAME
    processor = AutoProcessor.from_pretrained(processor_src, token=True)

    kwargs = dict(
        device_map={"": 0},
        dtype=_DTYPE,
        low_cpu_mem_usage=True,
        token=True,
    )
    if quantize_4bit:
        kwargs["quantization_config"] = build_quant_config(skip_vision_quant)

    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, **kwargs)

    if has_adapter:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return processor, model


def build_prompt(processor, image: Image.Image, prompt_text: str = PROMPT_TEXT) -> str:
    """Chat-formatted prompt string for one image (already contains ``<bos>``)."""
    user_msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    return processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)


def build_inputs(processor, image: Image.Image, prompt_text: str = PROMPT_TEXT):
    """Processor inputs for one image + prompt, on CPU.

    ``add_special_tokens=False`` because ``build_prompt`` already emitted
    ``<bos>`` (PLAN 4, C3). Callers move the batch to ``model.device``.
    """
    prompt = build_prompt(processor, image, prompt_text)
    return processor(text=[prompt], images=[image], add_special_tokens=False, return_tensors="pt")


def decode_new_tokens(processor, generated_ids, input_len: int) -> str:
    """Decode only the newly generated tokens (never the prompt) and strip any
    leaked chat-template boilerplate as a safety net (PLAN 4, Instruction 18)."""
    text = processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)
    return strip_template_leak(text).strip()


def strip_template_leak(text: str) -> str:
    """Remove chat-template boilerplate if a decode ever leaks it."""
    for marker in ("system\n", "user\n", "model\n"):
        idx = text.find(marker)
        if idx != -1 and idx < 40:
            text = text[idx + len(marker):]
    return text


def generate_report_for_image(processor, model, image: Image.Image,
                              prompt_text: str = PROMPT_TEXT) -> str:
    """One-image greedy generation using the shared settings."""
    inputs = build_inputs(processor, image, prompt_text)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, **GENERATE_KWARGS)
    return decode_new_tokens(processor, out, input_len)


def load_image(path: str | Path) -> Image.Image:
    """Open an image robustly (handles Windows backslash paths from the CSVs)."""
    with Image.open(Path(str(path).replace("\\", "/"))) as im:
        return im.convert("RGB")
