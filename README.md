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

## Reference results (real-only 7-pathology test split, 2026-09-03)

| Metric | Value |
|---|---|
| Test AUC (micro) | 0.8024 |
| Test F1 (micro)  | 0.3548 |

Model of record: `checkpoints/base_best_model.pth`.

## Quick start

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
