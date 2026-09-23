#!/usr/bin/env python3
"""PLAN 4 Phase 4.0 — measure before changing anything.

Loads the model the same way ``finetune.py`` does, then runs N micro-steps on
real samples and reports:

* seconds per micro-step (median after warm-up),
* ``torch.cuda.max_memory_allocated()``,
* time split: collator (CPU) vs forward+backward (GPU),
* trainable parameters grouped by prefix (confirms/refutes S1),
* processor image size, and the first token ids of a training example
  (confirms C3 double-<bos> and C4 <end_of_turn> ids).

Results are written to ``results/plan4/profile_before.json`` (default) so the
"before" row exists for every later speed claim (Instruction 1/2).

Usage:
    python scripts/profile_finetune.py --steps 30 --limit 64
    python scripts/profile_finetune.py --vision-lora          # old S1 behaviour
    python scripts/profile_finetune.py --quantize-vision      # old S3 behaviour
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor
# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from peft import get_peft_model

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "generation"))
from generation.medgemma_io import MODEL_NAME, PROMPT_TEXT, load_model_and_processor  # noqa: E402
from generation.finetune import (  # noqa: E402
    LORA_TARGET_REGEX,
    LEGACY_TARGET_MODULES,
    ReportCollator,
    build_lora_config,
    trainable_params_by_prefix,
)

TRAIN_CSV = "data/report_splits/train.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile MedGemma fine-tuning micro-steps.")
    ap.add_argument("--steps", type=int, default=30, help="Micro-steps to time.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=64, help="Number of train rows to use.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--vision-lora", action="store_true",
                    help="Attach LoRA to the vision tower too (old S1 behaviour).")
    ap.add_argument("--quantize-vision", action="store_true",
                    help="Also 4-bit quantise the vision tower (old S3 behaviour).")
    ap.add_argument("--output", default="results/plan4/profile_before.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    import transformers
    import bitsandbytes  # pyrefly: ignore [missing-import]
    import peft
    versions = {
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "torch": torch.__version__,
    }
    print(f"versions: {versions}")

    # -------------------- Instruction 2: processor/token checks ------------
    processor = AutoProcessor.from_pretrained(MODEL_NAME, token=True)
    image_size = processor.image_processor.size
    print(f"processor.image_processor.size = {image_size}")
    tok = processor.tokenizer
    special_ids = [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id]
    print(f"special tokens: bos={tok.bos_token_id} eos={tok.eos_token_id} pad={tok.pad_token_id}")
    print(f"convert_ids_to_tokens([1, 106]) = {tok.convert_ids_to_tokens([1, 106])}")

    # -------------------- Model & LoRA -------------------------------------
    model = load_model_and_processor(
        adapter_dir=None,
        quantize_4bit=True,
        skip_vision_quant=not args.quantize_vision,
    )[1]
    lora_cfg = build_lora_config(restrict_to_lm=not args.vision_lora)
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Confirm the module path contains `language_model` (Instruction 3 sketch).
    example_paths = [n for n, _ in model.named_modules()
                     if "layers.0.self_attn.q_proj" in n]
    print(f"sample q_proj paths: {example_paths[:4]}")
    prefix_counts = trainable_params_by_prefix(model)
    print(f"trainable params by prefix: {prefix_counts}")

    # -------------------- Data & collator ----------------------------------
    ds = load_dataset("csv", data_files={"train": TRAIN_CSV})["train"]
    ds = ds.select(range(min(args.limit, len(ds))))
    collator = ReportCollator(processor, PROMPT_TEXT)

    # Token check on a real training example (C3): first tokens should contain
    # exactly one <bos>.
    first_ids = collator._one(ds[0])[0]
    first_tokens = tok.convert_ids_to_tokens(first_ids[:5].tolist())
    print(f"first 5 tokens of a training example: {first_tokens}")
    bos_count = sum(1 for t in first_tokens if t in ("<bos>", "<s>"))

    # -------------------- Timed micro-steps --------------------------------
    collator_ms, fwd_bwd_ms, all_ms = [], [], []
    peak_mem = 0
    for step in range(args.steps):
        feat = [ds[step % len(ds)]]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = collator(feat)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        out = model(**batch, use_cache=False)
        loss = out.loss
        if step >= args.warmup:  # warm-up: compile/cudnn/disk cache effects
            collator_ms.append((t1 - t0) * 1000)
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if step >= args.warmup:
            fwd_bwd_ms.append((t2 - t1) * 1000)
            all_ms.append((t2 - t0) * 1000)
        model.zero_grad(set_to_none=True)
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        if step < args.warmup or step % 10 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}  "
                  f"collate={(t1 - t0) * 1000:.0f}ms  fwd+bwd={(t2 - t1) * 1000:.0f}ms")

    def med(v):
        return sorted(v)[len(v) // 2] / 1000 if v else None

    report = {
        "mode": "baseline(dd428b6-equivalent)" if args.vision_lora else "plan4(restricted-lm)",
        "vision_lora": args.vision_lora,
        "quantize_vision": args.quantize_vision,
        "batch_size": args.batch_size,
        "versions": versions,
        "image_size": image_size,
        "first_5_tokens": first_tokens,
        "bos_count_in_prompt": bos_count,
        "special_tokens": {k: v for k, v in zip(["bos", "eos", "pad"], special_ids)},
        "end_of_turn_id": tok.convert_tokens_to_ids("<end_of_turn>"),
        "trainable_params_by_prefix": prefix_counts,
        "sec_per_step_median": round(med(all_ms), 4),
        "collator_sec_median": round(med(collator_ms), 4),
        "fwd_bwd_sec_median": round(med(fwd_bwd_ms), 4),
        "peak_vram_gb": round(peak_mem / 1024**3, 3),
        "steps_timed": len(all_ms),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== PROFILE SUMMARY ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
