#!/usr/bin/env python3
"""Fine‑tune the MedGemma multimodal model (image‑text‑to‑text) on the IU‑Xray dataset.

This script mirrors the original Qwen2‑VL fine‑tuning script but swaps in the
MedGemma model class (``AutoModelForImageTextToText``) and processor. It uses
LoRA (via ``peft``) with 4‑bit quantisation (``bitsandbytes``) to keep VRAM
requirements low (≈6 GB on a RTX 4050).

Training steps
--------------
1. Load the MedGemma processor and 4‑bit‑quantised base model.
2. Attach a LoRA adapter and enable gradient checkpointing (see note below).
3. Load CSV splits containing ``image_path`` and ``report_text`` columns.
4. Use a custom collator that builds a chat‑style prompt (image + static
   ``PROMPT_TEXT``) and masks the prompt tokens from the language‑model loss.
5. Run ``Trainer`` with early‑stopping on the validation split.
6. Save the LoRA‑adapted model and processor for inference.

The script can be launched with ``python src/generation/finetune.py`` after
installing the required dependencies (see the instructions below).
"""

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
# pyrefly: ignore [missing-import]
from peft import LoraConfig, get_peft_model
# pyrefly: ignore [missing-import]
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration – edit these paths / hyper‑parameters as needed
# ---------------------------------------------------------------------------
MODEL_NAME = "google/medgemma-1.5-4b-it"  # HuggingFace repo name for MedGemma
TRAIN_CSV = "data/report_splits/train.csv"
VALIDATION_CSV = "data/report_splits/validation.csv"
OUTPUT_DIR = "checkpoints/medgemma_finetune"
BATCH_SIZE = 1               # Per‑GPU batch size (1 for 4‑bit + LoRA)
GRAD_ACCUM_STEPS = 8         # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
EPOCHS = 15                  # Upper bound – early stopping will usually stop earlier
LEARNING_RATE = 1e-4
LORA_R = 32
LORA_ALPHA = 32
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 3
SEED = 42

# Prompt used for every training example – must match the one used at inference
PROMPT_TEXT = "Write a chest X-ray report with Findings and Impression."


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for MedGemma fine‑tuning, but none was detected.")
    torch.cuda.empty_cache()
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------
    # Load processor & base model (4‑bit quantisation)
    # -----------------------------------------------------
    processor = AutoProcessor.from_pretrained(MODEL_NAME,token=True,trust_remote_code=True)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 is MedGemma's native dtype (RTX 4050 supports it)
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_cfg,
        device_map={"": 0},  # force onto GPU 0
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=True,  # <-- same token handling
        trust_remote_code=True,  # <-- keep if the model defines custom classes
    )
    model.config.use_cache = False  # KV cache is useless (and harmful) during training

    # -----------------------------------------------------
    # LoRA configuration
    # -----------------------------------------------------
    # NOTE: do NOT use prepare_model_for_kbit_training here. It upcasts every
    # fp16 param to fp32, which on medgemma-1.5-4b-it includes the ~262k-vocab
    # embedding (1.3 GB → 2.7 GB) and leaves a 6 GB GPU with 0 bytes free, so
    # the first forward pass OOMs. Gradient checkpointing + input grads give
    # the same benefit (frozen-input grads + activation checkpointing) without
    # the fp32 blow-up.
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Canonical PEFT order: enable checkpointing AFTER wrapping with the adapter,
    # then require input grads so the frozen first layer receives gradients.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # -----------------------------------------------------
    # Load CSV datasets – keep raw columns for the collator
    # -----------------------------------------------------
    datasets = load_dataset(
        "csv", data_files={"train": TRAIN_CSV, "validation": VALIDATION_CSV}
    )
    train_dataset = datasets["train"]
    validation_dataset = datasets["validation"]

    # -----------------------------------------------------
    # Custom collator that builds a chat‑style example and masks the prompt
    # -----------------------------------------------------
    class ReportCollator:
        """Create a single training example (BATCH_SIZE == 1).

        The collator:
        1. Loads the image from ``image_path``.
        2. Builds a user message consisting of the image and ``PROMPT_TEXT``.
        3. Applies the processor chat template to obtain ``prompt_text``.
        4. Concatenates ``prompt_text`` and the ground‑truth report, then tokenises.
        5. Masks the prompt tokens (label ``-100``) so LoRA only learns to generate
           the report text.
        """

        def __call__(self, features):
            if len(features) != 1:
                raise ValueError("ReportCollator expects BATCH_SIZE == 1")

            feature = features[0]
            image_path = feature["image_path"]
            report = str(feature["report_text"]).strip()
            if not report:
                raise ValueError(f"Empty report text for {image_path}")

            with Image.open(image_path) as raw_img:
                image = raw_img.convert("RGB")

            user_msg = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            }]

            prompt_text = processor.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=True
            )

            eos_token = processor.tokenizer.eos_token or ""
            full_text = f"{prompt_text}{report}{eos_token}"

            # Tokenise prompt and full sequence separately – needed for loss mask
            prompt_inputs = processor(
                text=[prompt_text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = processor(
                text=[full_text], images=[image], padding=True, return_tensors="pt"
            )
            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            # Mask prompt tokens
            labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
            inputs["labels"] = labels

            # Return CPU tensors: the Trainer's DataLoader pins CPU tensors, then
            # moves the batch to the model's device itself. Returning CUDA tensors
            # here would crash pin_memory ("only dense CPU tensors can be pinned").
            return inputs

    data_collator = ReportCollator()

    # -----------------------------------------------------
    # Training arguments – similar to the original Qwen2 script
    # -----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        fp16=False,
        bf16=True,   # RTX 4050 supports bf16; matches bnb_4bit_compute_dtype
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["labels"],
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    # Save the LoRA‑adapted checkpoint and processor for later inference
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuning complete - checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
