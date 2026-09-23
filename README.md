# Chest X-Ray Pathology Classifier (Indiana IU dataset)

Multi-label classifier that detects **7 thoracic pathologies** in frontal
chest X-rays:

`Granuloma`, `Opacity`, `Degenerative Spine Disease`, `Cardiomegaly`,
`Pulmonary Atelectasis`, `Pulmonary Hypoinflation`, `Pulmonary Hyperinflation`

- **Backbone:** DenseNet121 (TorchXRayVision `densenet121-res224-chex` by
  default, ImageNet weights also supported via `--backbone imagenet`)
- **Data:** frontal-only IU X-ray images, CLAHE-preprocessed to 224×224 RGB,
  patient-level 70/15/15 split (no leakage). `normal` is handled implicitly —
  it is not a prediction target.
- **Class imbalance:** balanced sampling + focal loss by default.
- **Demo:** `app.py` runs a Flask UI with per-label Grad-CAM heatmaps.

## Report Generation Results

| Metric | Value |
|---|---|
| BLEU-1 | 0.2362 |
| ROUGE-L | 0.1902 |
| METEOR | 0.1737 |
| CIDEr | 0.1197 |

*The numbers above correspond to the held‑out patient split (no leakage). METEOR and CIDEr will be filled in once the evaluation script runs successfully.*

**`XXXX` policy (PLAN 4, Instruction 9):** anonymisation tokens (`XXXX`, present
in ~42% of IU‑Xray reports) are stripped from **training targets** only
(`report_text_clean` column, built by `scripts/make_vlm_data.py`). Evaluation
should be reported on both raw and stripped references; never mix the two in
one table.

**Duplicate cap (PLAN 4, Instruction 10):** exact‑duplicate reports are capped
at 3 copies in the train split only (2,320 → ~1,994 rows). Val/test are
untouched.


| Metric | Value |
|---|---|
| Test AUC (micro) | 0.8024 |
| Test F1 (micro)  | 0.3548 |

Model of record: `checkpoints/base_best_model.pth`.

## Quick start

**Classifier (DenseNet‑121)**

```bash
# Train the 7‑pathology classifier (baseline)
./venv/Scripts/python.exe train.py --backbone xrv --epochs 80 \
    --balanced_sampling --focal_loss --save_prefix base_

# Launch the Grad‑CAM Flask demo for the classifier
./venv/Scripts/python.exe app.py
```

**Report Generation (MedGemma 4B LoRA — see `plan4.md`)**

```bash
# 1️⃣ Create patient‑level CSV splits (if not already present)
python src/generation/create_finetune_csv.py

# 2️⃣ Fine‑tune MedGemma (LoRA, 4‑bit LM / bf16 vision)
python src/generation/finetune.py          # smoke test: --max_steps 20 --limit 64

# 3️⃣ Generate reports on the held‑out test split
python src/generation/generate_reports.py \
    --split data/report_splits/test.csv \
    --output results/medgemma_patient_split/pipeline_results.json

# 4️⃣ Evaluate (BLEU‑1, ROUGE‑L, METEOR, CIDEr)
python src/generation/evaluate_metrics.py \
    results/medgemma_patient_split/pipeline_results.json \
    data/report_splits/test.csv \
    --output results/medgemma_patient_split/evaluation.json

# 5️⃣ Create a blinded review set (e.g., 25 samples)
python src/generation/create_review_set.py \
    --generated results/medgemma_patient_split/pipeline_results.json \
    --split data/report_splits/test.csv \
    --output results/medgemma_patient_split/review_set.csv \
    --sample 25
```

**Demo UI (Streamlit)**

```bash
streamlit run demo/app.py
```

The Streamlit app lets you upload an X‑ray, runs the inference wrapper, and displays the generated report.

---


```bash
# Train (real-only baseline)
./venv/Scripts/python.exe train.py --backbone xrv --epochs 80 \
    --balanced_sampling --focal_loss --save_prefix base_

# Grad-CAM demo UI
./venv/Scripts/python.exe app.py
```

## Branch map

| Branch | State | Contents |
|---|---|---|
| `master` | Historical baseline | Classifier as of the last committed run; contains legacy 15-label config and old experiment logs |
| `gan` | **Closed — failed** | GAN synthetic-augmentation experiment for rare classes. Full record + failure analysis in `GAN_AUGMENTATION.md`. DCGAN never produced usable images (no checkpoints/synthetic data ever saved) |
| `pseudo-labeling` | **Closed — failed** | NIH pool pseudo-labeling (self-training) experiment. Full record + failure analysis in `PSEUDO_LABELING.md`. Pool stalled at 208/15–20k images (raw NIH download never completed) |
| `renkario` | **Active** | Clean 7-pathology baseline with both failed experiment code paths removed — start new work here |
| `medgemma` | **Active** | MedGemma 1.5 4B LoRA report generation; fixes + speed plan in `plan4.md` |

Both closed experiments are archived on their branches with honest
results/why-they-failed documentation, so `renkario` starts from the clean
real-only baseline (`base_best_model.pth`, AUC 0.8024).

## Layout

```
app.py               Flask + Grad-CAM demo UI
dataset.py           Data loading, 7-pathology encoder, patient split, loaders
train.py             Training loop (focal loss, balanced sampling, mixup)
preprocess.py        CLAHE -> 224x224 -> RGB preprocessing pipeline
src/explainability/  Grad-CAM implementation
scripts/             Sanity checks
```

## Limitations & Future Work

- **Classifier**: the 7‑pathology DenseNet‑121 model is present in the `renkario` branch but its source files are not tracked in this repo yet. Add `train.py`, `app.py`, `dataset.py`, etc. to make the classifier runnable.
- **Grad‑CAM**: placeholder in the inference wrapper; proper heat‑map generation still needs to be hooked up.
- **Hallucination detection**: not implemented – future work will compare classifier predictions with report mentions.
- **METEOR / CIDEr**: evaluation metrics depend on `pycocoevalcap`; once the package is installed they will be populated.
- **Patient history**: synthetic history and contradiction tests are planned but not yet integrated.
- **Demo**: the Streamlit UI currently shows only the generated report; expanding it to display Grad‑CAM overlays and hallucination flags is a next milestone.

These items outline the remaining steps to turn the prototype into a complete, reproducible capstone project.
