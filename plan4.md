# PLAN 4 — Fix the MedGemma Fine-Tuning Script (Correctness first, then Speed)

Branch: `medgemma` · Commit reviewed: `dd428b6` ("messedup with the finetune script")
Hardware: RTX 4050 laptop, 6 GB VRAM, Windows
Stack (from `requirements.txt`): transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2, torch 2.13+cu126

---

## 0. How this diagnosis was done (and what it cannot prove)

| Level | Meaning | How I got it |
|---|---|---|
| **CONFIRMED** | Read directly in the repo or in the pinned library source | Cloned the branch; downloaded `transformers==5.16.1` and read `modeling_gemma3.py`, `processing_gemma3.py`, `modeling_siglip.py`, `modeling_utils.py`, `training_args.py` |
| **MEASURED** | Computed from the repo's own CSVs / result JSONs | Ran scripts on `data/report_splits/*.csv` and `results/*/pipeline_results.json` |
| **ESTIMATE** | Reasoned, not timed | I have no GPU here and `google/medgemma-1.5-4b-it` is gated, so nothing was actually trained. Every seconds-per-step number below is an estimate until Phase 4.0 measures it |

I also do not have your error output. The commit message says the script is "messed up" but not *how* it fails. This plan covers every failure the code itself makes likely. If you still hit a specific traceback after Phase 4.1, paste it (see §7).

---

## 1. Executive summary

There are two separate problems and they need separate fixes.

**Problem A — it is slow.** The biggest single cause is that the LoRA config attaches adapters to the *vision tower* as well as the language model, so every training sample back-propagates through a 27-layer, ~4,096-patch image encoder. On top of that: the image encoder is quantised to 4-bit, 224 px images are upscaled to the model's 896 px input, the collator does all CPU work twice on the main thread, the loss materialises full 262k-vocab fp32 logits, and the schedule allows up to 15 epochs with early-stopping patience 3.

**Problem B — it is not working properly.** Three things:

1. **Inference is still wired to Qwen.** `generate_reports.py`, `pipeline.py`, `demo/app.py`, and the scripts all load `Qwen2VLForConditionalGeneration` and `checkpoints/qwen_finetune_patient_split`. The MedGemma script saves to `checkpoints/medgemma_finetune`. A finished MedGemma run therefore cannot be evaluated or demoed.
2. **The previous fine-tuned outputs do not look image-conditioned.** In the Qwen run, 493 test images produced only **8 distinct reports**. In `finetuned_patient_split`, they produced 128 distinct reports, and **one report was emitted 93 times**. A constant "same normal report for every image" baseline scores at or above the existing model on ROUGE-L and CIDEr (§3, C2). Lower training loss alone will not prove MedGemma fixes this; we need an explicit image-dependence test.
3. **Several tokenisation details are wrong or fragile** (double `<bos>`, non-native end-of-turn token, collator that cannot be used with DataLoader workers on Windows).

**Order of work:** correctness fixes (4.1) → measure (4.0/4.2) → speed fixes (4.3) → image-dependence gate (4.5) → only then the long training run (4.6).

---

## 2. Why it is taking so long

### 2.1 The arithmetic (this alone explains "so much time")

- 2,320 train images, `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=8` → 2,320 forward/backward passes per epoch (290 optimizer steps).
- `EPOCHS=15`, `EARLY_STOPPING_PATIENCE=3` → the run can **never stop before epoch 4** (best epoch + 3 non-improving epochs), and can go to 15.
- Validation runs every epoch on 494 images.

Wall-clock as a function of seconds per sample `s` (train pass; eval pass assumed ≈ s/3):

| s (sec/sample) | 1 epoch (train) | Earliest possible stop (4 ep + 4 evals) | Full 15 epochs |
|---|---|---|---|
| 2 | 1.3 h | ~5.5 h | ~19 h |
| 4 | 2.6 h | ~11 h | ~39 h |
| 6 | 3.9 h | ~16.5 h | ~58 h |

**ESTIMATE:** with the current script on a 4050, `s` is plausibly in the 3–6 s range. Phase 4.0 measures the real number.

### 2.2 Ranked causes of a high `s`

**S1 — LoRA is attached to the vision tower. (CONFIRMED)**
`target_modules` is the name list `q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj`. PEFT matches these by module-name suffix anywhere in the model. In `modeling_siglip.py` the SigLIP attention layers are literally `q_proj`, `k_proj`, `v_proj`, `out_proj` (lines 268–271). So LoRA is injected into `q/k/v` of all 27 vision layers.
`modeling_utils.py:3205` shows gradient checkpointing defaults to `use_reentrant=False`, so gradients do flow into those vision adapters. Every sample therefore pays: vision forward + vision recompute + vision backward, for a 4,096-token sequence. That is far more tokens than the ~350 the language model sees (256 image tokens + prompt + ~50 report tokens).

**S2 — 224 px images are upscaled to 896 px. (CONFIRMED input size; model-side size to verify)**
`preprocess.py` writes 224×224 CLAHE images for DenseNet. Gemma3-family processors resize to the vision tower's native size (896 for MedGemma; verify with `print(processor.image_processor.size)`). The encoder therefore runs a full 64×64-patch pass on an image that only contains 16×16 patches' worth of information. This is both the most expensive input size *and* a fidelity loss.

**S3 — The vision tower is 4-bit quantised. (CONFIRMED by design of bitsandbytes; effect to measure)**
`BitsAndBytesConfig` has no `llm_int8_skip_modules`, so every `nn.Linear` in SigLIP is NF4-quantised and dequantised on every forward. That costs time and likely degrades the visual features the language model depends on.

**S4 — CPU work is serial and duplicated. (CONFIRMED)**
- `dataloader_num_workers` is not set → default 0 (`training_args.py:601`). PNG decode + resize-to-896 + normalise runs on the main thread between GPU steps.
- The collator calls `processor(...)` twice per sample (once for `prompt_inputs`, once for `inputs`), each time re-processing the same image.
- `ReportCollator` is defined *inside* `main()`. On Windows (spawn), a locally defined class cannot be pickled, so simply setting `num_workers>0` would crash. This is what blocks the obvious fix.

**S5 — Full-vocabulary logits. (CONFIRMED)**
`Gemma3ForConditionalGeneration.forward` (lines 1022–1041) upcasts the *entire* `logits` tensor to float32 when `labels` are given. Vocab is ~262k, sequence ~350 → ~370 MB per copy, with shifted/contiguous copies and the backward buffer on top. Only ~50 positions (the report) carry loss. On a 6 GB card this is both a memory and a time cost.

**S6 — Schedule and data. (MEASURED)**
- 577 of 2,320 train reports (25%) are exact repeats of an earlier report; the five most common reports alone account for ~150 samples. More epochs mostly re-teach the same sentences.
- The cosine schedule is stretched over 15 epochs (4,350 steps). If early stopping fires at epoch 5–6, LR never decays, which is a poor way to end a run.
- `eval_strategy="epoch"` over 494 images is a large fixed cost per epoch.

---

## 3. Why it is "not working properly"

**C1 — Inference path is Qwen-only. (CONFIRMED)**
Files that still hard-code Qwen: `src/generation/generate_reports.py`, `src/inference/pipeline.py`, `scripts/adapter_probe.py`, `scripts/sanity_gen_test.py`, `scripts/demo_test.py`, `demo/app.py`, plus `README.md` / `execution_summary.txt`. The last commit changed prompt strings and metrics in these files but did not swap the model class, processor, base-model id, or checkpoint dir. Also note the commit dropped classifier grounding from the prompt, but `generate_reports.py` still runs the classifier for every image (fine if you still need `predicted_findings` for the hallucination checker; wasteful otherwise).

**C2 — Existing fine-tuned outputs are near-constant. (MEASURED)**

| Result file | Test images | Distinct generated reports | Most common report repeated |
|---|---|---|---|
| `qwen_finetune_patient_split` | 493 | **8** | — |
| `finetuned_patient_split` | 493 | **128** | **93×** |

Quick scoring on `data/report_splits/test.csv` (lower-cased, `pycocoevalcap` without the PTB tokenizer, so absolute numbers differ slightly from the repo's `evaluation.json`; compare rows to each other only):

| System | BLEU-1 | ROUGE-L | CIDEr |
|---|---|---|---|
| Existing fine-tuned results | 0.267 | 0.217 | 0.120 |
| **Constant baseline:** the single most common train report, output for every image | 0.291 | 0.221 | 0.300 |

A model that ignores the image and prints the most common report matches or beats the current system on all three. So "loss went down" and "BLEU is non-zero" are not evidence of learning. The fix plan includes an explicit gate (4.5).
Why this happens on IU-Xray: ~15% of reports are near-templated "no acute…" text, 25% are exact duplicates, and 224 px CLAHE input gives weak signal, so the cheapest way to lower loss is to learn the language prior.

**C3 — Probable double `<bos>`. (CONFIRMED mechanism; token check needed)**
`apply_chat_template(..., tokenize=False)` returns text that already begins with `<bos>`. `processor(text=..., images=...)` then tokenises it again with the tokenizer's default `add_special_tokens=True`, which prepends a second `<bos>`. Training and inference both do this today, so they are at least consistent, but it is off-distribution for the base model. Verify with Phase 4.0 (`tokenizer.convert_ids_to_tokens(ids[:3])`).

**C4 — Wrong end-of-sequence token for the chat format. (CONFIRMED)**
The target is `report + processor.tokenizer.eos_token`. Gemma's assistant turn ends with `<end_of_turn>`. It works at inference only because the generation config lists both ids as stop tokens. Train on `<end_of_turn>\n` so the fine-tuned model is on-template.

**C5 — Training images are a mismatch for a VLM. (CONFIRMED)**
The VLM is fed the DenseNet-oriented 224 px CLAHE images. It should be fed the highest-resolution source image (letterboxed to square), which costs the same GPU time at 896 px and carries real detail.

**C6 — Data targets contain anonymisation tokens. (MEASURED)**
41.6% of train reports contain `XXXX`. The model will learn to emit `XXXX`, which hurts readability and any downstream keyword-based hallucination check, and inflates n-gram scores artificially on other `XXXX` references. Pick one policy and apply it to training targets *and* evaluation references.

**C7 — Smaller fragilities. (CONFIRMED)**
- `torch_dtype=` is deprecated in transformers 5.x (`modeling_utils.py:1491` warns) → use `dtype=`.
- `trust_remote_code=True` is unnecessary for a natively supported Gemma3 architecture and only widens the attack surface; drop it. `token=True` is fine.
- `model.config.use_cache = False` may not reach the nested `text_config`; harmless once gradient checkpointing is on, but don't rely on it.
- `image_path` values in the CSVs use Windows backslashes (`data\images\preprocessed\...`). They work on your machine but nowhere else; normalise with `Path(...)`.
- `requirements.txt` is UTF-16 LE. pip tolerates the BOM, but GitHub and most diff tools show it as binary. Convert to UTF-8.
- Commit `8abfb4f` shows the author already hit a 6 GB OOM with `prepare_model_for_kbit_training`; the workaround in `dd428b6` is reasonable and should be kept (see 4.1).

---

## 3b. Things that are already fine (don't touch)

- Patient-level split: uid overlap between train/val/test is 0 (MEASURED).
- Prompt-masking logic: `labels[:, :prompt_len] = -100` is correct as long as prompt and full text are tokenised identically (they are).
- `token_type_ids` is returned by the Gemma3 processor by default (`processing_gemma3.py:27,77–78`), so image tokens get bidirectional attention. The collator must keep passing it through.
- `bf16=True` + `bnb_4bit_compute_dtype=torch.bfloat16` is consistent.
- Not calling `prepare_model_for_kbit_training` (fp32 upcast of the 262k-vocab embedding) is right for 6 GB.

---

## 4. The fix plan

Each phase has **Instruction → Why → Checkpoint**, same format as `PLAN_B.md`. Do them in order; do not start 4.6 until 4.5 passes.

### Phase 4.0 — Measure before changing anything (30 minutes)

**Instruction 1.** Add `scripts/profile_finetune.py` that loads the model exactly as `finetune.py` does, runs 30 micro-steps on real samples, and prints: seconds/step (median after 5 warm-up steps), `torch.cuda.max_memory_allocated()`, time spent in `collator` vs `forward+backward` (wrap each in `time.perf_counter()` with `torch.cuda.synchronize()`), and the number of trainable parameters **grouped by prefix** (`vision_tower`, `multi_modal_projector`, `language_model`).
**Why.** This turns §2 from an estimate into numbers, and gives the "before" row for every later speed claim. The trainable-parameters-by-prefix line directly confirms or refutes S1.

**Instruction 2.** In the same script print: `processor.image_processor.size`, the first 5 token ids of a training example decoded with `convert_ids_to_tokens`, and `processor.tokenizer.convert_ids_to_tokens([1, 106])`.
**Why.** Confirms S2 (896?), C3 (two `<bos>`?), and C4 (ids of `<eos>`/`<end_of_turn>`).

**Checkpoint:** a saved `results/plan4/profile_before.json` with s/step, peak VRAM, dataloader-vs-GPU split, and trainable-param counts by prefix.

---

### Phase 4.1 — Correctness fixes in `finetune.py` (P0)

**Instruction 3 — restrict LoRA to the language model.** Replace the list with a regex (PEFT treats a string as a full-match regex):

```python
lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
)
model = get_peft_model(model, lora_cfg)
bad = [n for n, p in model.named_parameters() if p.requires_grad and "vision" in n]
assert not bad, f"LoRA leaked into vision tower: {bad[:3]}"
model.print_trainable_parameters()
```

*(Sketch — print `[n for n,_ in model.named_modules() if "layers.0.self_attn.q_proj" in n]` once to confirm the path contains `language_model`; adjust the regex if the v5 naming differs.)*

**Why.** Removes S1, the largest cost. With no trainable parameter upstream of the language model, the vision tower runs forward-only and stores no activations.

**Instruction 4 — fix tokenisation.**
- Pass `add_special_tokens=False` to every `processor(...)` call that receives text from `apply_chat_template(tokenize=False)`.
- Target = `report + "<end_of_turn>\n"` instead of `eos_token`.
- Use `gradient_checkpointing_kwargs={"use_reentrant": False}` explicitly and keep `enable_input_require_grads()`.

**Why.** C3 and C4. Explicit checkpointing kwargs protect you if the library default changes again.

**Instruction 5 — module-level, single-pass collator.** Move `ReportCollator` out of `main()`, give it `processor` and `prompt_text` in `__init__`, process the image **once** for the prompt, tokenise the report separately, and concatenate:

```python
class ReportCollator:
    def __init__(self, processor, prompt_text):
        self.p, self.prompt_text = processor, prompt_text
        self.pad = processor.tokenizer.pad_token_id

    def _one(self, f):
        with Image.open(Path(str(f["image_path"]).replace("\\", "/"))) as im:
            image = im.convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt_text}]}]
        prompt = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pi = self.p(text=[prompt], images=[image], add_special_tokens=False, return_tensors="pt")
        rep = self.p.tokenizer(str(f["report_text"]).strip() + "<end_of_turn>\n",
                               add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([pi["input_ids"], rep], 1)
        tt  = torch.cat([pi["token_type_ids"], torch.zeros_like(rep)], 1)
        lab = torch.cat([torch.full_like(pi["input_ids"], -100), rep], 1)
        return ids[0], tt[0], lab[0], pi["pixel_values"][0]

    def __call__(self, feats):
        rows = [self._one(f) for f in feats]
        L = max(r[0].numel() for r in rows)
        def pad(t, v): return torch.nn.functional.pad(t, (0, L - t.numel()), value=v)
        return {
            "input_ids":      torch.stack([pad(r[0], self.pad) for r in rows]),
            "attention_mask": torch.stack([pad(torch.ones_like(r[0]), 0) for r in rows]),
            "token_type_ids": torch.stack([pad(r[1], 0) for r in rows]),
            "labels":         torch.stack([pad(r[2], -100) for r in rows]),
            "pixel_values":   torch.stack([r[3] for r in rows]),
        }
```

**Why.** Halves CPU work (S4), makes `BATCH_SIZE > 1` and `dataloader_num_workers > 0` possible on Windows (top-level class is picklable), and keeps `token_type_ids` correct for the bidirectional image mask. Right-padding is correct for training.

**Instruction 6 — API hygiene.** `torch_dtype=` → `dtype=`; remove `trust_remote_code`; add `WARNING` if `processor.image_processor.size` is not 896.

**Instruction 7 — wire inference to MedGemma (fixes C1).** Create `src/generation/medgemma_io.py` with `load_model_and_processor()`, `build_inputs(processor, image)`, and `decode_new_tokens(...)`. `finetune.py`, `generate_reports.py`, `pipeline.py`, `scripts/*`, and `demo/app.py` must all import the **same** prompt text and input-building function. Base model = `MODEL_NAME`, adapter = `checkpoints/medgemma_finetune`, `AutoModelForImageTextToText`, `dtype=torch.bfloat16`. Keep `input_len` slicing on decode.
**Why.** The train/inference prompt drift bug (`"…Findings"` vs `"…Findings and Impression."`) that `dd428b6` patched by hand cannot recur if there is only one definition. Also delete or clearly mark the Qwen checkpoints as legacy.

**Checkpoint:** `python src/generation/finetune.py --max_steps 20` runs without error; the trainable-params-by-prefix line shows **zero** `vision_tower` entries; the decoded first tokens show a single `<bos>`; `generate_reports.py --split ... --limit 5` loads the MedGemma adapter and returns text.

---

### Phase 4.2 — Data and target hygiene (P0/P1, ~1 hour)

**Instruction 8 — VLM-quality images.** Regenerate VLM inputs from the *original* PNGs, not the 224 px CLAHE files: letterbox to a square (keep aspect ratio), resize to 896 with LANCZOS, no CLAHE (or a mild one), save to `data/images/vlm/`. Point `report_splits/*.csv` at a new `vlm_image_path` column. Keep the 224 px folder for the DenseNet classifier.
**Why.** Same GPU cost as today's upscaled inputs (S2) but real detail (C5). Doing it offline also removes the per-step resize cost.

**Instruction 9 — decide the `XXXX` policy and freeze it.** Recommended: strip `XXXX` tokens from training targets and, separately, report metrics on both raw references and `XXXX`-stripped references. Write the choice into the README so numbers stay comparable.
**Why.** C6.

**Instruction 10 — de-duplicate the prior.** Cap any exact-duplicate report at 3 copies in the *train* split only (val/test untouched). MEASURED on the current CSV: 2,320 → 1,994 rows at a cap of 3 (−14%); a cap of 5 only gets to 2,081 (−10%), so 3 is the useful setting.
**Why.** Reduces the pull toward the template report (C2) and trims epoch time by ~14% for free. This is a modest gain, not the main speed fix.

**Checkpoint:** new CSVs written; a one-line script prints train/val/test row counts, duplicate counts, and the fraction of reports containing `XXXX` after cleaning.

---

### Phase 4.3 — Speed fixes (P1, apply one at a time and re-run the profiler)

Apply in this order; keep each only if `s/step` improves and peak VRAM stays under ~5.5 GB.

**Instruction 11 — keep the vision path out of 4-bit.**

```python
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
)
```

**Why.** Removes dequant overhead in the vision encoder and preserves its features (S3). **Trade-off:** ~0.5–0.7 GB more VRAM (**ESTIMATE**). If it OOMs, revert and rely on Instruction 12 instead.

**Instruction 12 — batch size and workers.** Try `BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=4`, then `BATCH_SIZE=4`, `GRAD_ACCUM_STEPS=2` (same effective batch of 8); set `dataloader_num_workers=2`, `dataloader_persistent_workers=True`. Because the vision tower is now forward-only, memory per sample is far smaller than today.
**Why.** At batch size 1 a 4-bit model is dominated by kernel-launch and dequant overhead; batching is close to a linear speed-up until VRAM runs out (S4).

**Instruction 13 — cheaper loss.** Subclass `Trainer.compute_loss`: run the model *without* `labels` and with `logits_to_keep=K` (`K` = longest report length + 1 in the batch, with **left**-padded batches so the report is always at the tail), then compute cross-entropy yourself on those K positions.
**Why.** Shrinks the 262k-vocab fp32 logits from ~350 positions to ~60 (S5). This is the most intricate change; do it only if the profiler still shows large memory/`lm_head` time after 11–12.

**Instruction 14 — right-size the schedule.**
- `EPOCHS = 3`, `EARLY_STOPPING_PATIENCE = 1` (cosine now decays fully inside the run).
- `eval_strategy="steps"`, `eval_steps` ≈ 150 optimizer steps, evaluating on a fixed **150-image validation subset** (full 494 only at the end).
- `save_strategy` matching, `save_total_limit=2`, `save_only_model=True`.

**Why.** S6. Small IU-Xray adapters typically fit in 2–4 epochs; more epochs just memorise the templated reports (C2).

**Instruction 15 (optional, largest possible win) — cache image features.** If the vision tower and projector are frozen, the 256×2560 image embeddings for each image never change. Pre-compute them once (~1.3 MB each in bf16, ~3–4 GB for all splits), then train the language model on `inputs_embeds` with image positions filled. This removes the vision forward entirely.
**Risk:** needs care to keep `token_type_ids` and image-token positions aligned with the model's masking path, and it forbids on-the-fly image augmentation. Treat as experimental; validate that loss on 20 samples matches the uncached path within noise before using it.

**Checkpoint:** `results/plan4/profile_after.json`. **Target (not a promise): ≤ 1 s/sample, i.e. under ~40 min per epoch and a complete 3-epoch run in roughly 2–3 hours.** If the profiler shows something else, use the measured number.

---

### Phase 4.4 — Image-dependence gate (must pass before the long run)

**Instruction 16.** After a **short** pilot (≈300 optimizer steps), generate on 100 fixed validation images with greedy decoding and compute:
1. **Distinct-output ratio** = distinct reports / 100.
2. **Shuffled-image test:** rerun the same 100 prompts with each image replaced by a *different* image's pixels; compute ROUGE-L/CIDEr for correct vs shuffled pairing.
3. **Constant-baseline comparison:** ROUGE-L/CIDEr of the pilot vs the constant most-common-report baseline (§3, C2) on the same 100 images.
4. **Finding agreement:** for 3–4 easy labels (cardiomegaly, effusion, opacity/consolidation, pneumothorax) do a keyword match in the generated vs reference text and report per-label precision/recall.

**Suggested pass criteria (adjust once, then freeze before seeing test data):**
- distinct-output ratio clearly above the old 8/493 and 128/493 (aim > 0.5);
- correct-image ROUGE-L/CIDEr beat shuffled-image by a visible margin;
- pilot beats the constant baseline on CIDEr;
- at least one abnormal finding predicted with precision above its base rate.

**Why.** C2. This gate costs minutes and prevents spending a day of GPU on a run that learned only the language prior. If it fails: (a) confirm 4.1's LoRA restriction and 4.2's image fix took effect; (b) unfreeze nothing else yet; (c) try a higher LR (2e-4) and `r=16, alpha=32`; (d) only then consider LoRA on the vision projector (`multi_modal_projector`) as a deliberate, measured experiment.

**Checkpoint:** `results/plan4/gate_report.json` with all four numbers.

---

### Phase 4.5 — The real run and re-evaluation

**Instruction 17.** Run the full training with the schedule from Instruction 14 and the best of the pilot settings. Save adapter + processor to `checkpoints/medgemma_finetune`.

**Instruction 18.** Regenerate the test split with `generate_reports.py`, evaluate with `evaluate_metrics.py` (keep the chat-template-leak stripper as a safety net), and report **all of**: BLEU-1/4, ROUGE-L, METEOR, CIDEr, distinct-output ratio, and the constant-baseline row, in one table. Never report BLEU/CIDEr without the constant-baseline row and the distinct-output ratio next to it.

**Instruction 19.** Only after this: rerun the hallucination checker and Grad-CAM pipeline on the new outputs, so every downstream number comes from the same generation run.

**Checkpoint:** `results/medgemma_patient_split/{pipeline_results,evaluation}.json` and a comparison table with rows: constant baseline / Qwen LoRA / MedGemma zero-shot / MedGemma LoRA.

---

### Phase 4.6 — Housekeeping

- Convert `requirements.txt` to UTF-8; add the `transformers`/`peft`/`bitsandbytes` versions used for the run to the training log header.
- Update `README.md`, `execution_summary.txt`, and `demo/app.py` title (currently "Qwen-2-VL-2B LoRA") to MedGemma; add `plan4.md` to the Branch map.
- Normalise CSV image paths to forward slashes (or store relative `Path` objects).
- Replace throwaway commit messages ("klsdfj", "messedup…") with what changed and why; it will save you when you need to bisect a regression.
- Add `--max_steps` and `--limit` flags to `finetune.py` / `generate_reports.py` so smoke tests are one command.

---

## 5. Priority / effort table

| # | Change | Fixes | Effort | Expected effect |
|---|---|---|---|---|
| 3 | LoRA → language model only | S1 | 10 min | Largest speed-up; less VRAM |
| 7 | Wire inference to MedGemma | C1 | 1–2 h | Makes results evaluable |
| 4–5 | Tokenisation + single-pass, picklable collator | C3, C4, S4 | 1 h | Correct inputs; enables workers/batching |
| 8 | 896 px letterboxed inputs from raw PNGs | S2, C5 | 1 h | Same cost, better signal |
| 16 | Image-dependence gate | C2 | 1 h | Prevents a wasted long run |
| 11–12 | Skip-quantise vision, batch + workers | S3, S4 | 30 min each | Moderate speed-up, verify VRAM |
| 14 | 3 epochs, subset eval | S6 | 10 min | Removes most of the fixed cost |
| 9–10 | `XXXX` policy, duplicate cap | C6, S6 | 1 h | Less prior-collapse, faster epochs |
| 13 | Loss on tail positions only | S5 | 2 h | Memory/time; only if still needed |
| 15 | Cached image features | S1–S3 | 1 day | Potentially biggest, riskiest |

## 6. Decisions you need to make

1. **`XXXX` policy** (strip vs keep) — affects comparability with published IU-Xray numbers.
2. **Reference text** — Findings-only vs Findings+Impression (the prompt asks for both; the CSV column is a single `report_text`). Whatever you choose must match between training targets, prompt wording, and evaluation references.
3. **Keep the classifier in the loop?** The last commit removed classifier grounding from the prompt. If it stays removed, the classifier now only feeds Grad-CAM and the hallucination check, and the README's grounding claims should be updated.

## 7. If it still fails, send me these

- The full traceback (or the last 40 log lines) and which phase it happened in.
- `nvidia-smi` output while training is running.
- `results/plan4/profile_before.json` / `profile_after.json`.
- The first log lines from `print_trainable_parameters()` and the by-prefix breakdown.

## 8. Known unknowns (so nothing here is oversold)

- I could not load `google/medgemma-1.5-4b-it` (gated, no GPU), so module paths under `language_model`, the processor's image size, and the token ids are inferred from the Gemma3 code in the pinned transformers release. Phase 4.0 verifies each in minutes.
- All timings are estimates until Phase 4.0 runs on your machine.
- The constant-baseline scores in §3 use my own quick scoring setup (no PTB tokenizer, lower-cased), so compare them to each other, not to your README table.
