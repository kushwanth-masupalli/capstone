"""
Chest X-Ray Image Preprocessing
================================
Applies CLAHE, resize, and grayscale-to-RGB conversion.
Run this ONCE before training. Do not run during training.

Input:  data/images/*.png (raw DICOM-converted PNGs)
Output: data/images/preprocessed/*.png (224x224 RGB, CLAHE-enhanced)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ─── CONFIG ──────────────────────────────────────────────────────────────────

SRC_DIR = Path("data/images")
DST_DIR = Path("data/images/preprocessed")

TARGET_SIZE = 224          # DenseNet121 input size
CLAHE_CLIP_LIMIT = 2.0     # Standard starting point
CLAHE_GRID_SIZE = 8        # Tile grid size for CLAHE


# ─── PREPROCESSING FUNCTION ──────────────────────────────────────────────────

def preprocess_image(src_path: Path, dst_path: Path) -> bool:
    """
    Apply CLAHE → resize → grayscale-to-RGB and save as PNG.
    
    Args:
        src_path: Path to raw input image
        dst_path: Path to save preprocessed image
    
    Returns:
        True if successful, False if image failed to load
    """
    # Load as grayscale (chest X-rays are single-channel)
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"  [SKIP] Failed to load: {src_path.name}")
        return False
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_GRID_SIZE, CLAHE_GRID_SIZE)
    )
    img = clahe.apply(img)
    
    # Resize to target dimensions
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Convert grayscale to 3-channel RGB
    # DenseNet121 expects 3-channel input
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Save preprocessed image
    cv2.imwrite(str(dst_path), img)
    return True


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    """Process all images in source directory."""
    
    # Validate source directory exists
    if not SRC_DIR.exists():
        print(f"ERROR: Source directory not found: {SRC_DIR}")
        print("Please check the path and try again.")
        return
    
    # Create output directory
    DST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files
    image_files = sorted(list(SRC_DIR.glob("*.png")))
    
    if len(image_files) == 0:
        print("ERROR: No PNG images found in source directory.")
        return
    
    print(f"Found {len(image_files)} images in {SRC_DIR}")
    print(f"Output directory: {DST_DIR}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"CLAHE settings: clipLimit={CLAHE_CLIP_LIMIT}, grid={CLAHE_GRID_SIZE}x{CLAHE_GRID_SIZE}")
    print()
    
    # Process each image with progress bar
    success_count = 0
    fail_count = 0
    
    for img_path in tqdm(image_files, desc="Preprocessing"):
        dst_path = DST_DIR / img_path.name
        
        # Skip if already processed (idempotent)
        if dst_path.exists():
            success_count += 1
            continue
        
        if preprocess_image(img_path, dst_path):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print()
    print("=" * 50)
    print(f"Preprocessing complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Output:     {DST_DIR}")


if __name__ == "__main__":
    main()
