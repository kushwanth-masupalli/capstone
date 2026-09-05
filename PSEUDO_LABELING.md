# Pseudo-Labeling (NIH Pool Expansion) — Experiment Record & Failure Analysis

> **Status: CLOSED — did not meet expectations (2026-09-05).** This branch
> archives the semi-supervised self-training experiment: what was tried, what
> the artifacts show, and why the approach was abandoned before it could be
> evaluated. The classifier code here is the real-only 7-pathology IU
> baseline plus the pseudo-label expansion path (`--pseudo_csv`).

---

## 1. Goal

The multi-label classifier on the frontal-only IU dataset is limited by a
small, imbalanced training set (~2.6k real images). The goal of this branch
was to **expand the train set with real, unlabeled NIH ChestXray14 images**,
pseudo-labeled by the existing IU classifier (the "teacher"), and beat the
teacher baseline on the untouched real-IU test split:

| Metric (test, pathology) | Teacher baseline (`checkpoints/base_best_model.pth`, 2026-09-03) |
|---|---|
| AUC (micro) | 0.8024 |
| F1 (micro)  | 0.3548 |
| F1 (macro)  | 0.4067 |

**Go/no-go bar (pre-agreed):** test AUC ≥ **0.8224** (0.8024 + 0.02) with
test F1 (micro) not below 0.3548. Val/test must contain only real,
radiologist-labeled IU images; pseudo-labels enter the train partition only.

## 2. Method (as designed)

1. `src/pseudo/prepare_pool.py` — preprocess 15–20k real NIH frontal images
   with the same CLAHE → 224×224 pipeline as IU → `data/pseudo_pool/`.
2. Teacher = the best real-only 7-pathology model (`base_best_model.pth`).
3. `src/pseudo/generate_pseudo_labels.py` — run the teacher over the pool,
   keep a class only above a **0.85** confidence threshold (per-class
   overrides allowed for rare classes, never a uniform lowering), write
   `data/pseudo_labels.csv` + raw-prob npz + per-class contact sheets for
   visual spot-check.
4. Retrain with `train.py --pseudo_csv data/pseudo_labels.csv` — pseudo rows
   drawn at 0.5× the weight of real rows (`build_train_sampler`).
5. `src/pseudo/evaluate_checkpoint.py` — judge against the go/no-go bar.

## 3. Evidence — what is actually on this branch / on disk

| Artifact | State | Meaning |
|---|---|---|
| `src/pseudo/` (prepare_pool, generate_pseudo_labels, evaluate_checkpoint) | Present | Full pipeline was implemented |
| `data/nih_raw/` | **Empty** | Raw NIH ChestXray14 archives never fully downloaded (or were removed) |
| `data/pseudo_pool/PP_*.png` | **208 images** (target: 15–20k) | Phase 1 preprocessing stalled almost immediately |
| `data/pseudo_pool/pool_manifest.csv` | **Missing** | The interrupted `prepare_pool` run never completed its manifest write |
| `data/pseudo_labels.csv`, `pseudo_probs.npz`, `_review/` contact sheets | **Missing** | Phase 3 (teacher inference + filtering) never ran |
| `checkpoints/pseudo_best_model.pth` | **Missing** | Phase 4 (expanded-data training) never ran |

## 4. Result

**The experiment never reached its go/no-go bar — or any measurable point.**
The pool acquisition stalled at **208 of the planned 15–20k images** and the
raw NIH source vanished (empty `data/nih_raw/`), so:

- no pseudo-labels were ever generated (`pseudo_labels.csv` does not exist),
- no pseudo-labeled model was ever trained (`pseudo_best_model.pth` does not
  exist),
- there is therefore no expanded-data AUC/F1 to compare against the 0.8024
  teacher baseline. Expectation **not met**.

## 5. Why it failed

1. **Data acquisition was the bottleneck.** NIH ChestXray14 is the largest
   public chest X-ray set (~112k images, ~42 GB across the `images_*.tar.gz`
   archives). The download never completed and the partial source was
   removed, leaving only **208 preprocessed pool images** — two orders of
   magnitude below the plan's 15–20k target, which already assumed a heavy
   confidence filter would discard most of the pool.
2. **The stall cascaded through the whole plan.** Phase 1 feeds everything:
   no pool of real NIH images → no teacher inference → no pseudo-labels →
   no expanded training set → nothing for Phase 4/5 to evaluate. The plan's
   own Phase 0 rule (real-only val/test) held, but so did the *real-only
   train*, so the model was never given the opportunity to improve.
3. **Even with a complete pool, the approach carried known, documented
   risks:** confirmation bias (the teacher reinforcing its own mistakes —
   mitigated only by the 0.85 threshold and 0.5 real/pseudo weight) and
   NIH↔IU domain shift (different equipment/population), expected to transfer
   worst for the subtle IU findings this experiment was meant to fix.

## 6. Verdict & lessons

- **The methodology is sound but the data prerequisite was not met.** The
  teacher + high-confidence threshold + down-weighted pseudo rows pipeline is
  intact in `src/pseudo/` and can be resumed from a *complete, already
  downloadable* NIH pool (or any large unlabeled frontal X-ray set).
- **Budget the download before starting.** ~42 GB of archives + preprocessing
  of 15–20k full-resolution PNGs is the real cost of this approach; without a
  reliable copy of NIH on disk, Phase 1 cannot complete.
- This experiment is superseded; the real-only baseline remains the model of
  record (`checkpoints/base_best_model.pth`, AUC 0.8024).

## 7. Reproduction (for completeness — only viable once NIH is fully on disk)

```bash
# Phase 1: build the pool (needs the full NIH download present first)
./venv/Scripts/python.exe -m src.pseudo.prepare_pool \
    --source data/nih_raw \
    --data_entry_csv data/nih_raw/Data_Entry_2017.csv \
    --max_images 20000

# Phase 3: teacher inference + confidence filtering (0.85, per-class overrides)
./venv/Scripts/python.exe -m src.pseudo.generate_pseudo_labels \
    --teacher checkpoints/base_best_model.pth

# Phase 4: expanded-data training (real IU + pseudo pool, pseudo at 0.5x weight)
./venv/Scripts/python.exe train.py --backbone xrv --epochs 80 \
    --balanced_sampling --focal_loss --save_prefix pseudo_ \
    --pseudo_csv data/pseudo_labels.csv

# Phase 5: judge vs the bar (AUC >= 0.8224, F1 >= 0.3548)
./venv/Scripts/python.exe -m src.pseudo.evaluate_checkpoint \
    --checkpoint checkpoints/pseudo_best_model.pth --split test
```
