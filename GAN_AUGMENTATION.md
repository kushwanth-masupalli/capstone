# GAN Synthetic Augmentation — Experiment Record & Failure Analysis

> **Status: CLOSED — did not meet expectations (2026-09-05).** This branch
> archives the experiment: what was tried, what the artifacts show, and why
> the approach was abandoned. The classifier code here is the real-only
> 7-pathology IU baseline plus the GAN augmentation path (`--synthetic_dir`).

---

## 1. Goal

The multi-label classifier trained on the full frontal-only IU dataset
(7 pathology labels, real-only split) is class-imbalanced: the three rarest
classes (Pulmonary Hyperinflation, Pulmonary Hypoinflation, Pulmonary
Atelectasis) have 117–203 train images and the worst per-class F1. The goal
of this branch was to **balance the rarest classes by synthesizing extra
training images with one DCGAN per class**, add them to the **train split
only**, and beat the real-only baseline without regressing overall AUC/F1.

**Reference baseline** (locked real-only split, `checkpoints/base_best_model.pth`,
evaluated 2026-09-03 with `src/pseudo/evaluate_checkpoint.py`):

| Metric (test, pathology) | Value |
|---|---|
| AUC (micro) | 0.8024 |
| F1 (micro)  | 0.3548 |

Go/no-go bar: targeted per-class F1 must improve **and** overall test AUC /
micro-F1 must not regress.

## 2. Method (as designed)

1. `src/gan/analyze_classes.py` — pick target classes from real train counts
   (rarest classes above the ~50–80 image floor).
2. `src/gan/train_gan.py` — train one DCGAN per target class on its real
   train-split images only (224×224, ~300 epochs, sample grids written to
   `runs/gan/<Class>/` to watch for mode collapse).
3. `src/gan/generate.py` — generate ~2–3× the real count per class into
   `data/synthetic/<Class>/`, then visually review the contact sheet.
4. Retrain the classifier with `train.py --synthetic_dir data/synthetic`
   (synthetic images enter **train only**; val/test stay real by construction).
5. Compare `gan_*` checkpoints vs the `base_*` baseline on the go/no-go bar.

## 3. Evidence — what is actually on this branch / on disk

| Artifact | State | Meaning |
|---|---|---|
| `src/gan/` (analyze_classes, train_gan, dcgan, generate) | Present | Full GAN pipeline was implemented |
| `runs/gan/<3 classes>/` | **Empty** — no `iter_*`/`epoch_*` sample grids | GAN training never reached even the first grid save |
| `checkpoints/gan/<3 classes>/` | **Empty** — no `generator.pth`/`config.json` | No generator checkpoint ever saved |
| `data/synthetic/<3 classes>/` | **Empty** — 0 images | No synthetic image ever passed the quality gate |
| `checkpoints/gan_best_model.pth` + `gan_last_model.pth` | Exist (2026-09-03) | Phase-4 classifier run happened, but with **0** added images |

## 4. Result

**No measurable improvement was possible or observed.** The Phase 2/3 gates
of the plan were never cleared:

- The DCGAN runs created their output folders but **never saved a checkpoint
  or a sample grid** → they crashed or were aborted very early.
- `data/synthetic/` therefore contains **0 images**, so the Phase-4 classifier
  retrain (`gan_best_model.pth`) added nothing to the training set — it is a
  rerun of the baseline by construction.

Expectation (targeted per-class F1 ↑, AUC/F1 ≥ baseline) **not met**.

## 5. Why it failed

1. **The DCGAN never completed a training run.** Folders were created, then
   nothing was written — the run died inside epoch 1. The most plausible
   cause is a CUDA out-of-memory: training happens at **224×224 with a
   default batch size of 32**, which is far beyond the ~6 GB VRAM of the
   available GPU (RTX 4050). A smaller batch/image size would be required.
2. **Even with enough memory, the setup was unlikely to converge.** A plain
   DCGAN trained on **117–203 real images per class** is far below the data
   needed for stable GAN training; mode collapse is the expected outcome,
   which is exactly why the plan's own quality gate required visual review
   of sample grids before generating anything.
3. **Because Phase 2/3 never cleared, Phase 4 was a no-op** — the classifier
   was retrained against an empty `data/synthetic/`, so there was never any
   signal to measure an improvement on.

The failure is therefore **pipeline-level (GAN never produced usable data)**
rather than a measured "synthetic images hurt the classifier" result — that
comparison was never reached.

## 6. Verdict & lessons

- **Do not retry a plain DCGAN at 224² on this hardware.** If class
  balancing is revisited, options with better evidence are: cheaper
  resampling/weighting (already partially in place via balanced sampling +
  focal loss), stronger real-image augmentation, or collecting more real
  data — not synthetic generation on ~150-image classes.
- **Gate-keeping worked as intended**: no garbage synthetic images ever
  leaked into training, and val/test stayed real-only throughout.
- This experiment is superseded; the real-only baseline remains the model of
  record (`checkpoints/base_best_model.pth`, AUC 0.8024).

## 7. Reproduction (for completeness — not recommended)

```bash
# Phase 1: pick target classes
./venv/Scripts/python.exe -m src.gan.analyze_classes

# Phase 2: DCGAN per class (needs far more VRAM than 6 GB at this image size,
# or a much smaller batch_size / image resolution)
./venv/Scripts/python.exe -m src.gan.train_gan --class_name "Pulmonary Hyperinflation" --epochs 300
./venv/Scripts/python.exe -m src.gan.train_gan --class_name "Pulmonary Hypoinflation" --epochs 300
./venv/Scripts/python.exe -m src.gan.train_gan --class_name "Pulmonary Atelectasis" --epochs 300

# Phase 3: generate + rebuild manifest/contact sheet
./venv/Scripts/python.exe -m src.gan.generate --class_name "Pulmonary Hyperinflation" --multiplier 3

# Phase 4: baseline, then GAN-augmented classifier (train-only synthetic)
./venv/Scripts/python.exe train.py --backbone xrv --epochs 80 --balanced_sampling --focal_loss --save_prefix base_
./venv/Scripts/python.exe train.py --backbone xrv --epochs 80 --balanced_sampling --focal_loss --save_prefix gan_ --synthetic_dir data/synthetic
```
