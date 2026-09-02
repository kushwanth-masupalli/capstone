"""
TorchXRayVision Sanity-Check Script
====================================
Phase 1 of the CXR-Domain-Pretrained Backbone plan.

Loads each candidate XRV checkpoint, prints model structure, feature dims,
and runs a forward pass on one IU image to confirm everything works.

Usage:
    python scripts/sanity_check_xrv.py
"""

import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from pathlib import Path

# --- 1. Check torchxrayvision import ---
print("=" * 60)
print("Phase 1: TorchXRayVision Sanity Check")
print("=" * 60)

import torchxrayvision as xrv
print(f"torchxrayvision version: {xrv.__version__}")
print(f"PyTorch version:         {torch.__version__}")
print(f"CUDA available:          {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:                     {torch.cuda.get_device_name(0)}")

# --- 2. Load candidate checkpoints ---
CANDIDATES = [
    "densenet121-res224-all",
    "densenet121-res224-chex",
    "densenet121-res224-nih",
]

for ckpt_name in CANDIDATES:
    print(f"\n{'-' * 50}")
    print(f"Loading: {ckpt_name}")
    print(f"{'-' * 50}")
    try:
        model = xrv.models.DenseNet(weights=ckpt_name)
        model.eval()
        print(f"  Pathologies ({len(model.pathologies)}): {model.pathologies}")
        print(f"  Output features: {model.classifier.in_features}")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,}")

        print(f"  Classifier: {model.classifier}")

        feature_children = list(model.features.children())
        print(f"  Features layers: {len(feature_children)}")
        for i, layer in enumerate(feature_children[-3:], start=len(feature_children)-3):
            print(f"    [{i}] {layer}")

        dummy = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        print(f"  Random input shape:  {dummy.shape}")
        print(f"  Output shape:        {out.shape}")
        print(f"  Output range:        [{out.min().item():.2f}, {out.max().item():.2f}]")

    except Exception as e:
        print(f"  ERROR loading {ckpt_name}: {e}")

# --- 3. Run on a real IU image ---
print(f"\n{'=' * 60}")
print("Forward pass on real IU image")
print(f"{'=' * 60}")

from PIL import Image

images_dir = Path("data/images/preprocessed")
sample_img = None
if images_dir.exists():
    imgs = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if imgs:
        sample_img = imgs[0]

if sample_img is None:
    print("  No preprocessed images found -- skipping real-image test.")
else:
    print(f"  Image: {sample_img.name}")

    pil_img = Image.open(sample_img).convert("L")
    print(f"  PIL size: {pil_img.size}, mode: {pil_img.mode}")

    img_np = np.array(pil_img).astype(np.float32)
    img_np = img_np[np.newaxis, ...]  # (1, H, W)

    img_normalized = xrv.datasets.normalize(img_np, maxval=255.0)
    print(f"  After XRV normalize: shape={img_normalized.shape}, "
          f"range=[{img_normalized.min():.1f}, {img_normalized.max():.1f}]")

    resizer = xrv.datasets.XRayResizer(224)
    img_resized = resizer(img_normalized)
    print(f"  After XRayResizer: shape={img_resized.shape}")

    cropper = xrv.datasets.XRayCenterCrop()
    img_cropped = cropper(img_resized)
    print(f"  After XRayCenterCrop: shape={img_cropped.shape}")

    img_tensor = torch.from_numpy(img_cropped).unsqueeze(0).float()
    print(f"  Final tensor shape: {img_tensor.shape}")
    print(f"  Final tensor dtype: {img_tensor.dtype}")
    print(f"  Final tensor range: [{img_tensor.min().item():.1f}, {img_tensor.max().item():.1f}]")

    best_ckpt = "densenet121-res224-chex"
    print(f"\n  Running forward pass through {best_ckpt}...")
    model = xrv.models.DenseNet(weights=best_ckpt)
    model.eval()

    with torch.no_grad():
        out = model(img_tensor)

    probs = torch.sigmoid(out)
    print(f"  Output shape: {out.shape}")
    print(f"  Predictions:")
    for i, name in enumerate(model.pathologies):
        p = probs[0, i].item()
        flag = " ***" if p > 0.5 else ""
        print(f"    {name:25s} {p:.4f}{flag}")

    print(f"\n  Testing feature extraction (for downstream use)...")
    try:
        feat = model.features(img_tensor)
        print(f"  After features: {feat.shape}")
        feat_relu = torch.nn.functional.relu(feat)
        feat_pool = torch.nn.functional.adaptive_avg_pool2d(feat_relu, (1, 1))
        feat_flat = torch.flatten(feat_pool, 1)
        print(f"  After pooling+flatten: {feat_flat.shape}")
        print(f"  Feature dim for new classifier: {feat_flat.shape[1]}")
    except Exception as e:
        print(f"  Feature extraction failed: {e}")

print(f"\n{'=' * 60}")
print("Sanity check complete!")
print(f"{'=' * 60}")
