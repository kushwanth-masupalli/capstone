#!/usr/bin/env python3
"""
Fine‑tune Qwen2‑VL‑2B with LoRA (4‑bit quantisation) on the IU‑Xray dataset.

This script follows the high‑level plan described in ``PLAN_B.md``:

* Load the Qwen2‑VL‑2B‑Instruct model in 4‑bit mode via ``bitsandbytes``.
* Attach a LoRA adapter (rank 16, alpha 16, dropout 0.05) to the language‑model
  projection layers.
* Train on a CSV that contains one row per training example with the pre‑processed
  image path and the corresponding report text.
* Use gradient accumulation to achieve an effective batch size of 8 on a GPU
  with limited VRAM (≈6 GB).
* Save the LoRA‑adapted model and its processor for later inference.

The script is deliberately minimal – you will likely need to adapt the data‑loading
logic (paths, column names) to match your environment.
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration (adjust as needed)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # HuggingFace repo name
DATASET_CSV = "data/train_finetune.csv"   # CSV with columns: uid, image_path, report_text
OUTPUT_DIR = "checkpoints/qwen_finetune"
BATCH_SIZE = 1               # Per‑GPU batch size (1 for 4‑bit + LoRA)
GRAD_ACCUM_STEPS = 8         # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
EPOCHS = 2
LEARNING_RATE = 1e-4
SEED = 42

def main() -> None:
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- Load processor & model (4‑bit) --------------------
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # -------------------- Prepare LoRA -----------------------------------
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
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

    # -------------------- Dataset ---------------------------------------
    # Expected CSV columns: uid, image_path, report_text
    dataset = load_dataset("csv", data_files=DATASET_CSV)

    def _preprocess(example):
        # ``processor`` expects a Pillow image; we provide the path and let it load.
        image_path = example["image_path"]
        report = example["report_text"]
        inputs = processor(images=image_path, text=report, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0),
        }

    tokenized = dataset.map(_preprocess, remove_columns=dataset.column_names)

    # -------------------- Training arguments ---------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_steps=0,  # checkpointing handled manually if desired
        evaluation_strategy="no",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"] if "train" in tokenized else tokenized["default"],
    )

    trainer.train()
    # Save both the adapted model and the processor for later inference.
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Finetuning complete – checkpoint saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
