# PLAN B — From Capstone to IEEE-Publishable: LoRA Fine-Tuning + Defensible Benchmarking

## What Plan B Is (and what it is not)

Plan A (`docs/PLAN.md`, phases 0–12) is the working capstone pipeline: frozen
Qwen2-VL-2B + classifier-driven RAG + history + dual hallucination checks.
It is complete and functional.

Plan B is the **upgrade path to a publishable paper**. It makes exactly ONE
architectural change (fine-tune the generator) and then spends its effort on
making the *comparison honest and the novelty provable*. The unique parts of
Plan A — hallucination detection, RAG grounding, patient history, Grad-CAM —
are the parts no published baseline has, and Plan B is built around proving
those, not around winning a BLEU race we cannot win fairly.

**Hardware reality (checked):** RTX 4050 laptop, 6 GB VRAM, "enough time".
A 7B LLM (R2GenGPT's Llama2-7B, XrayGPT's Vicuna-7B) cannot be fine-tuned on
this GPU. Qwen2-VL-2B CAN — with 4-bit QLoRA. All training steps below are
sized for 6 GB.

**Why Plan B exists instead of "just copy R2GenGPT/PromptMRG":**
- R2GenGPT's win comes from *training on the dataset* — we must adopt that
  idea (train a light adapter on a frozen backbone), but with a 2B model that
  fits our GPU.
- PromptMRG's win comes from classifier-driven generation — we already have
  that via classifier→retrieval; rebuilding its encoder-decoder from scratch
  would discard all of Phase 7–11.
- XrayGPT needs a 7B fine-tune we cannot afford and outputs summaries, not
  Findings/Impression structure.

---

## Phase B0 — Lock the Comparison Protocol (do this FIRST, before any training)

**Instruction 1.** Reproduce the exact evaluation harness the literature uses:
the **R2Gen 7:1:2 patient split** of IU-Xray (train 70% / val 10% / test 20%),
with the **COCO-caption metric protocol** (BLEU-1/2/3/4, ROUGE-L, METEOR,
CIDEr) and the reference = the full report text (Findings + Impression), the
way R2GenGPT and PromptMRG report it.
**Why.** Every published number you will compare against (R2GenGPT BLEU-1
0.466–0.488, PromptMRG, METransformer, R2Gen) is measured on this split with
this protocol. Your current `report_quality.json` numbers (BLEU-1 0.169,
ROUGE-L 0.164) use a different split (8:1:1), Findings-only references, and a
different tokenizer — they are NOT comparable to any published table. Without
this step, no claim you make is defensible and reviewers will reject on this
alone. Reuse the ReXrank harness (https://rexrank.ai/) rather than writing
metrics from scratch — metric implementation differences are a classic
rejection reason.

**Instruction 2.** Before touching any model, run the B0 harness on your
CURRENT frozen pipeline (Phase 11 outputs) so you have a frozen-model baseline
in the correct protocol.
**Why.** This is your "before" number. Every later claim ("fine-tuning gains
X") needs this exact baseline. It also tells you the true starting gap to the
published baselines, which decides how you frame the paper.

**Checkpoint:** One script that, given a `generated_reports.json`, emits a
table in the R2Gen/COCO protocol; you have the frozen-model row filled in.
If this isn't reproducible, nothing else in Plan B is.

---

## Phase B1 — LoRA Fine-Tune Qwen2-VL-2B on the IU-Xray Train Split

**Instruction 3.** Write `src/generation/finetune.py`: load Qwen2-VL-2B-
Instruct in **4-bit** (bitsandbytes QLoRA), attach LoRA to the language-model
projections (rank 8–16, alpha 16, dropout 0.05), and fine-tune ONLY the LoRA
adapters. Input = image (448², same as inference) + the SAME `build_prompt()`
output used at inference (RAG chunks + history + instruction). Target = the
ground-truth Findings text of the train reports.
**Why.** This is R2GenGPT's "delta tuning" recipe transplanted onto a model
small enough for 6 GB. Keeping the exact same prompt format at train and
inference time is critical: if the model trains on one prompt shape and
generates with another, quality collapses. 4-bit + LoRA means trainable
parameters are a few million — the only thing that fits and converges on a
4050.

**Instruction 4.** Train hyperparameters: 5–10 epochs, batch size 1–2 with
gradient accumulation to an effective batch of 8, learning rate 1e-4
(AdamW), warmup 3%, cosine decay. Save checkpoints every epoch; keep the one
with the best val-set report loss (or best val BLEU-1 if loss plateaus).
**Why.** ~2,800 train reports × 5–10 epochs ≈ 14–28k steps is a realistic
budget for a 4050 (roughly overnight-to-a-few-days with "enough time") and is
the range where R2GenGPT-style adapters converge on IU-Xray. Tracking val
loss/BLEU per epoch tells you when the small dataset starts overfitting (very
likely after ~6–8 epochs — that's expected, pick the best checkpoint, don't
train longer).

**Instruction 5.** Run the **context ablation with the fine-tuned model**:
for a sample of test images, generate with (a) correct RAG context,
(b) deliberately wrong RAG context, (c) no context, and compare outputs.
**Why.** This proves the model still *uses* the retrieved context after
fine-tuning (it's known that training on reports can make a model ignore
extra context). It also produces an ablation table for the paper: "fine-tuned
model is grounded in RAG; frozen model is not (or vice versa)". The
`context_ablation.py` script already exists for the frozen model — rerun it.

**Checkpoint:** Val BLEU-1 improved meaningfully over the frozen baseline
(expected: from ~0.17 to somewhere in the 0.3–0.45 range — R2GenGPT territory).
If there is no improvement after 5 epochs, debug the prompt/reference mismatch
before spending more GPU time.

---

## Phase B2 — Regenerate, Re-Evaluate, Ablate

**Instruction 6.** Regenerate the full test set with the fine-tuned model
through the unmodified Phase 11 pipeline (`run_pipeline.py`), producing
`results/phase11/pipeline_results.json` (fine-tuned) and a matching
`generated_reports.json`.
**Why.** Every downstream number (hallucination flags, edge cases, Grad-CAM)
must come from the SAME pipeline run, or your paper's numbers are internally
inconsistent. Keep the frozen-model outputs archived — you need both for the
comparison table.

**Instruction 7.** Produce the full result matrix:
- Report quality (B0 protocol): frozen vs fine-tuned, both with RAG.
- Ablations: fine-tuned ± RAG, ± history (extend `ablation_rag.py` /
  `ablation_history.py` to the fine-tuned model).
- Classifier metrics (Phase 3/9) unchanged — they feed retrieval.
- Hallucination metrics (Phase 8/9) on the fine-tuned outputs.
**Why.** Reviewers require to see *what each component contributes*.
The ablation columns are also your honest evidence for the "why not just
fine-tune?" question: if fine-tuning reduces hallucination flags, that's a
headline result; if it increases them, that's still a publishable finding
(trade-off analysis), but you must know which before writing.

**Checkpoint:** One results JSON per experiment, all in the same protocol.
No number in the paper should exist that isn't regenerable from a saved
artifact.

---

## Phase B3 — Bulletproof the Novelty Axis (this is your paper's core)

**Instruction 8.** Strengthen the hallucination evaluation from a demo feature
into a metric: expand the manually reviewed sample to 40–60 reports, report
flag-level **precision, recall, and F1** for both check types, with two
reviewers (or one reviewer + one radiology-savvy friend) and inter-rater
agreement if possible. Add per-finding breakdowns (which findings hallucinate
most).
**Why.** No published baseline (R2GenGPT, PromptMRG, XrayGPT, METransformer)
reports ANY hallucination/safety metric. A rigorous, manually-validated
safety evaluation on a standard dataset is a publishable contribution even
when BLEU is mid-pack — it is the table only you can produce.

**Instruction 9.** Add clinical efficacy scoring: label generated + ground-
truth reports with the CheXpert labeler (or a simple trained labeler) and
report precision/recall/F1 across the 14 thoracic categories, the way
R2GenGPT's Table 2 does.
**Why.** Lexical metrics (BLEU/ROUGE) are known to be weak proxies for
clinical correctness — reviewers know this and reward clinical metrics.
R2GenGPT reports clinical F1 0.389 (deep) on MIMIC-CXR; you can report the
same quantity on IU-Xray for both your frozen and fine-tuned models.

**Instruction 10.** Write the failure-analysis section: 5–8 case studies
with image + Grad-CAM + generated report + hallucination flags, including at
least one true positive, one false positive, and one missed hallucination.
**Why.** Qualitative case studies are what make a systems paper credible and
readable. They also let you discuss the precision/recall trade-off honestly —
a sign of maturity reviewers reward.

**Checkpoint:** You can state, with numbers and cases: "our system detects
X% of hallucinations at Y% precision, and here is exactly where it fails."

---

## Phase B4 — Paper Write-Up & Positioning

**Instruction 11.** Frame the paper as a **system + safety** contribution, NOT
a "we beat SOTA" claim. Structure:
1. Introduction (the hallucination problem in LLM report generation)
2. Related work (R2Gen, R2GenGPT, PromptMRG, XrayGPT — cite from
   `docs/related_work_research.txt`)
3. Method (your pipeline: classifier → retrieval → history → VLM → dual
   hallucination check → Grad-CAM; note the one LoRA training step)
4. Experiments (B0 protocol table + ablations + clinical efficacy +
   hallucination metrics + case studies)
5. Limitations (single dataset, simulated history, rule-based checks)
6. Conclusion
**Why.** An honest "competitive on quality, unique on safety, zero-training
option" framing survives review; a "we beat R2GenGPT" framing fails the
moment a reviewer reruns your numbers. Your contribution is the *integrated
system with the first hallucination-safety evaluation on IU-Xray* — that is
novel regardless of BLEU rank.

**Instruction 12.** Put the comparison table LAST in the experiments section,
framed as "quality vs safety trade-off across system designs", with your
frozen row, your fine-tuned row, and the published rows (cited, marked † like
the literature does).
**Why.** Same-split comparison satisfies reviewers while the framing keeps
you out of the "why didn't you win" trap.

**Checkpoint:** A complete draft where every table row has a regenerable
artifact, and a one-paragraph "contribution" statement that does not contain
the word "state-of-the-art".

---

## Publication Reality Check (your question: IEEE paper, not a "simple" journal?)

**Short answer: yes, it is publishable — at an IEEE conference, realistically,
not a top IEEE journal. Here is the honest map.**

- **Realistic targets (IEEE Xplore-indexed conferences):** EMBC (IEEE
  Engineering in Medicine & Biology), BHI (IEEE Biomedical & Health
  Informatics), ISBI, BIBM, CBMS, ICIP, ICHI. These accept solid systems
  papers with a clear contribution; a single well-executed dataset +
  a genuinely new safety evaluation is enough. 4–8 pages, peer-reviewed,
  IEEE Xplore. This is a real publication and counts for most programs.
- **Stretch (needs more work):** IEEE JBHI (Journal of Biomedical and Health
  Informatics) and IEEE TMI (Medical Imaging). These are top-tier; for them
  you would additionally need (a) MIMIC-CXR results (PhysioNet credentialed)
  to show generality beyond one small dataset, (b) a trained/evaluated
  hallucination detector with statistical rigor, and (c) ideally a second
  contribution (e.g., the synthetic-history contradiction benchmarks from
  Phase 6 formalized as a public test set).
- **The "I can't show it beats previous works" fear is real but misframed.**
  Reviewers do NOT require beating SOTA on BLEU/ROUGE for a systems paper —
  but they DO require (1) same-split, same-protocol comparison, (2) honest
  trade-off analysis, and (3) a contribution no prior work has. You have #3
  (hallucination-safety evaluation + RAG/history grounding + zero-training
  mode, all in one system). What kills papers is exactly what Plan B fixes:
  non-comparable numbers and no safety metrics.
- **What would make it "not simple":** the hallucination evaluation (B3),
  the full ablation matrix (B2), and the clinical-efficacy table (B9). A
  paper that is "pipeline + BLEU table" is a workshop paper. A paper that is
  "pipeline + same-protocol comparison + safety metrics with manual
  validation + ablations + failure analysis" is an EMBC/BHI paper.

---

## Checklist (before any submission)

- [ ] B0 harness reproduces R2Gen/COCO-protocol numbers from saved artifacts
- [ ] Frozen-model row measured in B0 protocol (before training)
- [ ] LoRA checkpoint with val-loss tracking; best checkpoint saved
- [ ] Fine-tuned full test-set run through unmodified Phase 11 pipeline
- [ ] Ablation matrix: ± fine-tuning, ± RAG, ± history (all in B0 protocol)
- [ ] Clinical efficacy (CheXpert-style) table for frozen + fine-tuned
- [ ] Hallucination precision/recall/F1 on 40–60 manually reviewed reports
- [ ] 5–8 failure case studies with Grad-CAM + flags
- [ ] Draft whose contribution statement avoids "state-of-the-art"
- [ ] Venue shortlist chosen (conference first; journal = post-MIMIC extension)

## Effort estimate (RTX 4050, "enough time")

- B0 harness + frozen baseline: 1–2 days
- B1 LoRA training: 1–4 days of GPU time (overnight runs)
- B2 regeneration + ablations: 2–4 days
- B3 manual review + clinical eval: 3–5 days (human time, not GPU)
- B4 draft: 3–5 days
- Total: roughly 2–3 focused weeks for a conference-ready paper.