#!/usr/bin/env python3
"""Fine-tune MedGemma (image-text-to-text) with LoRA on the IU-Xray dataset.

Rewritten per PLAN 4 (plan4.md). Key changes vs the old script:

* LoRA is restricted to the **language model** with a regex (Instruction 3) -
  the old name-list config also injected adapters into the 27-layer SigLIP
  vision tower, so every sample back-propagated through ~4,096 vision tokens
  (PLAN 4, S1 - the single largest cost).
* Single-pass, module-level collator: the image is processed once,
  ``add_special_tokens=False`` avoids the double ``<bos>`` (C3), the target
  ends with ``<end_of_turn>`` (C4), and the class is picklable so
  ``dataloader_num_workers > 0`` works on Windows (S4).
* The vision tower / projector / lm_head stay in bf16 - only the language
  model is 4-bit quantised (Instruction 11).
* Right-sized schedule: 3 epochs, eval every 150 optimizer steps on a fixed
  150-image validation subset (Instruction 14).
* Shared prompt/model/tokenisation come from ``medgemma_io.py`` so training
  and inference cannot drift apart (Instruction 7).

Smoke test:
    python src/generation/finetune.py --max_steps 20 --limit 64 --eval-subset 16
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
# pyrefly: ignore [missing-import]
from peft import LoraConfig, get_peft_model
# pyrefly: ignore [missing-import]
from datasets import load_dataset

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # put `src` on the path
from generation.medgemma_io import (  # noqa: E402
    ADAPTER_DIR,
    END_OF_TURN,
    MODEL_NAME,
    PROMPT_TEXT,
    build_quant_config,
    load_model_and_processor,
)

# ---------------------------------------------------------------------------
# Configuration (defaults; overridable via CLI flags)
# ---------------------------------------------------------------------------
TRAIN_CSV = "data/report_splits/train.csv"
VALIDATION_CSV = "data/report_splits/validation.csv"
OUTPUT_DIR = str(ADAPTER_DIR)

BATCH_SIZE = 2               # Instruction 12: try 2x4, then 4x2 (effective batch 8)
GRAD_ACCUM_STEPS = 4
EPOCHS = 3                   # Instruction 14: cosine fully decays inside the run
LEARNING_RATE = 1e-4
LORA_R = 32
LORA_ALPHA = 32
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 1  # Instruction 14
EVAL_STEPS = 150             # Instruction 14: eval every ~150 optimizer steps
EVAL_SUBSET = 150            # fixed validation subset; full 494 only at the end
NUM_WORKERS = 2
SEED = 42

# Regex (PEFT treats a str target_modules as a full-match regex) - language
# model only. SigLIP uses q/k/v/out_proj too, which the old name-list config
# matched by accident and attached vision adapters (PLAN 4, S1).
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

# Old behaviour (S1), kept only for profiler comparisons: --vision-lora
LEGACY_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "down_proj", "up_proj"]


def build_lora_config(restrict_to_lm: bool = True) -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGET_REGEX if restrict_to_lm else LEGACY_TARGET_MODULES,
    )


def trainable_params_by_prefix(model) -> dict:
    """Count trainable parameters grouped by model part (PLAN 4, Instruction 1)."""
    groups = {"vision_tower": 0, "multi_modal_projector": 0, "language_model": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for key in groups:
            if key in name:
                groups[key] += p.numel()
                break
        else:
            groups["other"] += p.numel()
    return groups


class ReportCollator:
    """Build a supervised training batch (module-level => picklable for DataLoader workers).

    Processes the image ONCE (the old collator ran the processor twice per
    sample), tokenises the report separately with ``add_special_tokens=False``
    and concatenates, so the prompt part keeps its image token_type_ids and
    the report part is plain text. Right-padding is correct for training.
    """

    def __init__(self, processor, prompt_text: str = PROMPT_TEXT):
        self.p = processor
        self.prompt_text = prompt_text
        self.pad_id = processor.tokenizer.pad_token_id

    def _one(self, f):
        # Phase 4.2 writes `vlm_image_path` (896px letterboxed raw PNGs);
        # fall back to the old `image_path` for legacy CSVs.
        raw_path = f.get("vlm_image_path") or f["image_path"]
        if not str(raw_path).strip():
            raw_path = f["image_path"]
        image_path = Path(str(raw_path).replace("\\", "/"))
        report = str(f["report_text_clean" if "report_text_clean" in f else "report_text"]).strip()
        if not report:
            raise ValueError(f"Empty report text for {image_path}")

        with Image.open(image_path) as im:
            image = im.convert("RGB")

        prompt = self.p.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt_text},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # add_special_tokens=False: the chat template already emitted <bos> (C3).
        pi = self.p(text=[prompt], images=[image], add_special_tokens=False,
                    return_tensors="pt")
        # Gemma's assistant turn ends with <end_of_turn>, not <eos> (C4).
        rep = self.p.tokenizer(report + END_OF_TURN, add_special_tokens=False,
                               return_tensors="pt")["input_ids"]

        ids = torch.cat([pi["input_ids"], rep], dim=1)
        if "token_type_ids" in pi:
            tt = torch.cat([pi["token_type_ids"], torch.zeros_like(rep)], dim=1)
        else:  # older processor versions may omit it; zeros == text behaviour
            tt = torch.zeros_like(ids)
        lab = torch.cat([torch.full_like(pi["input_ids"], -100), rep], dim=1)
        return ids[0], tt[0], lab[0], pi["pixel_values"][0]

    def __call__(self, feats):
        rows = [self._one(f) for f in feats]
        seq_len = max(r[0].numel() for r in rows)

        def pad(t, v):
            return torch.nn.functional.pad(t, (0, seq_len - t.numel()), value=v)

        return {
            "input_ids": torch.stack([pad(r[0], self.pad_id) for r in rows]),
            "attention_mask": torch.stack([pad(torch.ones_like(r[0]), 0) for r in rows]),
            "token_type_ids": torch.stack([pad(r[1], 0) for r in rows]),
            "labels": torch.stack([pad(r[2], -100) for r in rows]),
            "pixel_values": torch.stack([r[3] for r in rows]),
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LoRA fine-tune MedGemma on IU-Xray.")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Run exactly N optimizer steps (smoke test). Overrides epochs.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N train rows (smoke test).")
    ap.add_argument("--eval-subset", type=int, default=EVAL_SUBSET,
                    help="Fixed validation subset size (0 = full validation split).")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--lora-r", type=int, default=LORA_R)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--quantize-vision", action="store_true",
                    help="Also 4-bit quantise the vision tower (old S3 behaviour).")
    ap.add_argument("--vision-lora", action="store_true",
                    help="Attach LoRA to the vision tower too (old S1 behaviour; "
                         "profiler comparisons only).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for MedGemma fine-tuning, but none was detected.")
    torch.cuda.empty_cache()
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    import transformers
    import bitsandbytes  # pyrefly: ignore [missing-import]
    import peft
    print(f"versions: transformers={transformers.__version__} peft={peft.__version__} "
          f"bitsandbytes={bitsandbytes.__version__} torch={torch.__version__}")

    # -------------------- Processor & base model (4-bit) --------------------
    processor = AutoProcessor.from_pretrained(MODEL_NAME, token=True)
    image_size = processor.image_processor.size
    print(f"processor.image_processor.size = {image_size}")
    if not (isinstance(image_size, int) and image_size >= 896
            or isinstance(image_size, dict) and min(image_size.values()) >= 896):
        print(f"WARNING: expected vision input >= 896px (MedGemma native), got {image_size}")

    model = load_model_and_processor(
        adapter_dir=None,  # training starts from the base model, no adapter
        quantize_4bit=True,
        skip_vision_quant=not args.quantize_vision,
    )[1]
    model.config.use_cache = False
    try:
        model.config.text_config.use_cache = False
    except AttributeError:
        pass

    # -------------------- LoRA (language model only) -----------------------
    lora_cfg = build_lora_config(restrict_to_lm=not args.vision_lora)
    model = get_peft_model(model, lora_cfg)
    if not args.vision_lora:
        leaked = [n for n, p in model.named_parameters()
                  if p.requires_grad and "vision" in n]
        assert not leaked, f"LoRA leaked into vision tower: {leaked[:3]}"
    model.print_trainable_parameters()
    print(f"trainable params by prefix: {trainable_params_by_prefix(model)}")

    # Canonical PEFT order: checkpointing AFTER wrapping, with explicit kwargs.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # -------------------- Dataset -----------------------------------------
    datasets = load_dataset("csv", data_files={"train": TRAIN_CSV, "validation": VALIDATION_CSV})
    train_dataset = datasets["train"]
    if args.limit:
        train_dataset = train_dataset.select(range(min(args.limit, len(train_dataset))))
    validation_dataset = datasets["validation"]
    if args.eval_subset and args.eval_subset < len(validation_dataset):
        validation_dataset = validation_dataset.select(range(args.eval_subset))
    print(f"train rows: {len(train_dataset)}  eval rows: {len(validation_dataset)}")

    data_collator = ReportCollator(processor, PROMPT_TEXT)

    # -------------------- Training arguments -------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        fp16=False,
        bf16=True,  # matches bnb_4bit_compute_dtype; RTX 4050 supports bf16
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,
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

    # Save the LoRA adapter + processor for inference (medgemma_io reads these).
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuning complete - checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
