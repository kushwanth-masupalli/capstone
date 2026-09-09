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
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
# pyrefly: ignore [missing-import]
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# pyrefly: ignore [missing-import]
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration (adjust as needed)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # HuggingFace repo name
TRAIN_CSV = "data/report_splits/train.csv"
VALIDATION_CSV = "data/report_splits/validation.csv"
OUTPUT_DIR = "checkpoints/qwen_finetune_patient_split"
BATCH_SIZE = 1               # Per‑GPU batch size (1 for 4‑bit + LoRA)
GRAD_ACCUM_STEPS = 8         # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
EPOCHS = 2
LEARNING_RATE = 1e-4
SEED = 42

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen2-VL fine-tuning, but none was detected.")
    torch.cuda.empty_cache()
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    # -------------------- Load processor & model (4‑bit) --------------------
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        # Force the trainable model onto the RTX 4050 instead of allowing
        # automatic CPU offloading, which makes the first batch appear stuck.
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
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
    # Keep raw paths/text in the dataset. Vision-language samples must be
    # processed at batch time, because image sizes and token lengths vary.
    datasets = load_dataset(
        "csv", data_files={"train": TRAIN_CSV, "validation": VALIDATION_CSV}
    )
    train_dataset = datasets["train"]
    validation_dataset = datasets["validation"]

    class ReportCollator:
        """Build one Qwen2-VL supervised training batch (BATCH_SIZE is one)."""

        def __call__(self, features):
            if len(features) != 1:
                raise ValueError("This collator supports BATCH_SIZE=1 only.")

            feature = features[0]
            image_path = feature["image_path"]
            report = str(feature["report_text"]).strip()
            if not report:
                raise ValueError(f"Empty report text for {image_path}")

            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")

            user_messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Write a chest X-ray report with Findings and Impression."},
                ],
            }]
            prompt_text = processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True
            )
            eos_token = processor.tokenizer.eos_token or ""
            full_text = f"{prompt_text}{report}{eos_token}"

            # Encode both versions with the same image.  Their length difference
            # lets us exclude instruction/image tokens from the language-model loss.
            prompt_inputs = processor(
                text=[prompt_text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = processor(
                text=[full_text], images=[image], padding=True, return_tensors="pt"
            )
            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels[:, :prompt_inputs["input_ids"].shape[1]] = -100
            inputs["labels"] = labels
            return inputs

    data_collator = ReportCollator()

    # -------------------- Training arguments ---------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # PeftModel hides the base model signature; explicitly identify the
        # supervised loss tensor so Trainer reports eval_loss during validation.
        label_names=["labels"],
        remove_unused_columns=False,
        # Avoid importing TensorBoard.  The local TensorFlow/TensorBoard pair is
        # incompatible, and logging is not required for this training run.
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    # Save both the adapted model and the processor for later inference.
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Finetuning complete – checkpoint saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
