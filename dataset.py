"""
Indiana Chest X-Ray Dataset
============================
PyTorch Dataset class for loading and preparing chest X-ray images.

Steps:
1. Merge indiana_reports.csv + indiana_projections.csv on 'uid'
2. Filter to Frontal (PA) images only
3. Parse the 'Problems' column into multi-label binary encoding
4. Load preprocessed images from data/images/preprocessed/
5. Apply ImageNet normalization (mean/std) on-the-fly

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
REPORTS_CSV = DATA_DIR / "indiana_reports.csv"
PROJECTIONS_CSV = DATA_DIR / "indiana_projections.csv"
IMAGES_DIR = DATA_DIR / "images" / "preprocessed"

# Multi-label: use top N most common individual problems
TOP_N_PROBLEMS = 15
MIN_SAMPLES = 10  # Minimum samples per problem to include

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Random seed for reproducibility
RANDOM_STATE = 42


# ─── DATA PREPARATION ────────────────────────────────────────────────────────

def load_and_merge_data():
    """
    Load both CSVs, merge on 'uid', filter to Frontal images only.
    
    Returns:
        pd.DataFrame with columns: uid, filename, projection, Problems, label_vector
    """
    reports = pd.read_csv(REPORTS_CSV)
    projections = pd.read_csv(PROJECTIONS_CSV)
    
    # Merge on uid
    merged = pd.merge(reports, projections, on="uid")
    
    # Filter to Frontal only
    frontal = merged[merged["projection"] == "Frontal"].copy()
    
    print(f"Loaded {len(merged)} total images, {len(frontal)} frontal images")
    print(f"Unique patients (frontal): {frontal['uid'].nunique()}")
    
    return frontal


# ─── PROBLEM NAME MAPPING ────────────────────────────────────────────────────
# Maps verbose/inconsistent names to clean, standardized labels
# This merges similar problems and normalizes naming

PROBLEM_NAME_MAP = {
    # Lung-related → merge into 'Lung'
    "Lung": "Lung",
    "Lung, Hyperlucent": "Lung",
    "Lung Diseases, Interstitial": "Lung",
    
    # Granuloma-related → merge into 'Granuloma'
    "Calcified Granuloma": "Granuloma",
    "Granuloma": "Granuloma",
    "Granulomatous Disease": "Granuloma",
    "Calcinosis": "Granuloma",
    
    # Vertebrae/spine-related → merge into 'Spine'
    "Thoracic Vertebrae": "Spine",
    "Spondylosis": "Spine",
    "Scoliosis": "Spine",
    
    # Aorta-related → merge into 'Aorta'
    "Aorta": "Aorta",
    "Aorta, Thoracic": "Aorta",
    "Atherosclerosis": "Aorta",
    
    # Airspace-related → merge into 'Opacity'
    "Opacity": "Opacity",
    "Airspace Disease": "Opacity",
    "Consolidation": "Opacity",
    
    # Other direct mappings (normalize names)
    "Cardiomegaly": "Cardiomegaly",
    "Cardiac Shadow": "Cardiomegaly",
    "Pulmonary Atelectasis": "Pulmonary Atelectasis",
    "Pleural Effusion": "Pleural Effusion",
    "Pulmonary Congestion": "Pulmonary Congestion",
    "Pulmonary Emphysema": "Pulmonary Emphysema",
    "Cicatrix": "Cicatrix",
    "Markings": "Markings",
    "Diaphragm": "Diaphragm",
    "Density": "Density",
    "Nodule": "Nodule",
    "Deformity": "Deformity",
    "Osteophyte": "Osteophyte",
    "Surgical Instruments": "Surgical Instruments",
    "Catheters, Indwelling": "Catheters",
    "Fractures, Bone": "Fractures",
    "Foreign Bodies": "Foreign Bodies",
    "Breast Implants": "Breast Implants",
}

# Noise labels to remove entirely
NOISE_LABELS = {
    "No Indexing",
    "Technical Quality of Image Unsatisfactory",
    "Tube, Inserted",
    "Implanted Medical Device",
}


def clean_problems(problems_str):
    """
    Clean a raw Problems string into standardized label names.
    
    Steps:
    1. Split by semicolon
    2. Remove noise labels
    3. Normalize names using PROBLEM_NAME_MAP
    4. Deduplicate
    
    Args:
        problems_str: raw string like "Calcified Granuloma;Calcified Granuloma"
        
    Returns:
        list of clean, deduplicated label names
    """
    if pd.isna(problems_str):
        return []
    
    # Split by semicolon, strip whitespace
    raw_labels = [p.strip() for p in str(problems_str).split(";") if p.strip()]
    
    # Remove noise and normalize
    clean_labels = []
    for label in raw_labels:
        # Skip noise
        if label in NOISE_LABELS:
            continue
        
        # Map to normalized name
        normalized = PROBLEM_NAME_MAP.get(label, None)
        
        # If not in map, try partial matching
        if normalized is None:
            for key, value in PROBLEM_NAME_MAP.items():
                if key in label:
                    normalized = value
                    break
        
        # Keep as-is if no mapping found (but still valid)
        if normalized is None:
            normalized = label
        
        clean_labels.append(normalized)
    
    # Deduplicate while preserving order
    seen = set()
    unique_labels = []
    for label in clean_labels:
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)
    
    return unique_labels


def build_label_encoder(df):
    """
    Parse 'Problems' column into multi-label binary encoding.
    
    The Problems column contains semicolon-separated values like:
        "Cardiomegaly;Pulmonary Congestion"
        "normal"
    
    Steps:
    1. Clean and normalize problem names
    2. Remove noise labels
    3. Deduplicate within each entry
    4. Count frequency and keep top N
    
    Args:
        df: DataFrame with 'Problems' column
        
    Returns:
        label_names: list of problem names (e.g., ['normal', 'Cardiomegaly', ...])
        label_vectors: np.ndarray of shape (n_samples, n_labels) with 0/1 values
    """
    # Clean and count problems
    problem_counter = Counter()
    all_problems = []
    
    for problems_str in df["Problems"]:
        clean_labels = clean_problems(problems_str)
        all_problems.append(clean_labels)
        problem_counter.update(clean_labels)
    
    # Filter by minimum sample count
    common_problems = [
        prob for prob, count in problem_counter.most_common()
        if count >= MIN_SAMPLES
    ]
    
    # Keep top N
    label_names = common_problems[:TOP_N_PROBLEMS]
    
    print(f"\nTop {TOP_N_PROBLEMS} problems (min {MIN_SAMPLES} samples):")
    for name in label_names:
        count = problem_counter[name]
        print(f"  {name}: {count} images")
    
    # Create binary label vectors
    label_vectors = np.zeros((len(df), len(label_names)), dtype=np.float32)
    
    for i, problems in enumerate(all_problems):
        for j, label_name in enumerate(label_names):
            if label_name in problems:
                label_vectors[i, j] = 1.0
    
    # Count how many images have at least one label
    has_label = (label_vectors.sum(axis=1) > 0).sum()
    print(f"\nImages with at least one label: {has_label}/{len(df)}")
    
    return label_names, label_vectors


# ─── TRANSFORMS ──────────────────────────────────────────────────────────────

def get_eval_transform():
    """
    Deterministic transform used for validation and test: just tensor + normalize.
    No augmentation, so metrics reflect the model's real behavior.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform():
    """
    Training-time augmentation.

    With only ~2.6k training images and a full-capacity pretrained backbone,
    the model overfits within a few epochs without augmentation (train loss
    keeps dropping while val loss climbs). These are deliberately mild —
    aggressive crops can crop pathology out of frame, and aggressive color
    jitter fights the CLAHE preprocessing already applied to these images.

    Horizontal flip is safe here because none of the 15 labels are
    laterality-specific (e.g. no "left" vs "right" pleural effusion) — the
    model doesn't need to learn a canonical orientation for any label used.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=10,             # small rotation
            translate=(0.05, 0.05),  # small shift
            scale=(0.95, 1.05),      # small zoom in/out
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── PYTORCH DATASET ─────────────────────────────────────────────────────────

class IndianaChestXRayDataset(Dataset):
    """
    PyTorch Dataset for Indiana Chest X-Ray images.
    
    Loads preprocessed images (224x224 RGB, CLAHE-enhanced) and applies
    ImageNet normalization on-the-fly.
    """
    
    def __init__(self, df, label_vectors, label_names, transform=None, train=False):
        """
        Args:
            df: DataFrame with 'filename' column
            label_vectors: np.ndarray of shape (n_samples, n_labels)
            label_names: list of problem names
            transform: torchvision transforms to apply. If None, falls back to
                get_train_transform() when train=True, else get_eval_transform().
            train: whether this split should get augmentation when transform
                is not explicitly given.
        """
        self.df = df.reset_index(drop=True)
        self.label_vectors = label_vectors
        self.label_names = label_names
        self.filenames = df["filename"].values

        if transform is None:
            self.transform = get_train_transform() if train else get_eval_transform()
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Load one image and its multi-label vector.
        
        Returns:
            image: torch.Tensor of shape (3, 224, 224), float32
            labels: torch.Tensor of shape (n_labels,), float32 (0/1)
        """
        filename = self.filenames[idx]
        img_path = IMAGES_DIR / filename
        
        # Load image as RGB
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms (ToTensor + ImageNet normalization)
        image = self.transform(image)
        
        # Get label vector
        labels = torch.tensor(self.label_vectors[idx], dtype=torch.float32)
        
        return image, labels


# ─── DATA LOADER FACTORY ─────────────────────────────────────────────────────

def get_dataloaders(batch_size=16, num_workers=0, test_size=0.15, val_size=0.15):
    """
    Create train, validation, and test DataLoaders.
    
    Split strategy:
        - 70% train, 15% validation, 15% test
        - Split by patient (uid) to avoid data leakage
        
    Args:
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        test_size: Fraction of data for testing
        val_size: Fraction of data for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load and merge data
    df = load_and_merge_data()
    
    # Build label encoder
    label_names, label_vectors = build_label_encoder(df)
    
    # Split by patient (uid) to prevent data leakage
    unique_uids = df["uid"].unique()
    
    # First split: train+val vs test
    trainval_uids, test_uids = train_test_split(
        unique_uids, test_size=test_size, random_state=RANDOM_STATE
    )
    
    # Second split: train vs val
    train_uids, val_uids = train_test_split(
        trainval_uids, test_size=val_size / (1 - test_size), random_state=RANDOM_STATE
    )
    
    # Create masks
    train_mask = df["uid"].isin(train_uids)
    val_mask = df["uid"].isin(val_uids)
    test_mask = df["uid"].isin(test_uids)
    
    print(f"\n=== Dataset Split ===")
    print(f"Train:      {train_mask.sum()} images ({train_uids.shape[0]} patients)")
    print(f"Validation: {val_mask.sum()} images ({val_uids.shape[0]} patients)")
    print(f"Test:       {test_mask.sum()} images ({test_uids.shape[0]} patients)")
    
    # Create datasets — only the training split gets augmentation; val/test
    # stay deterministic so the metrics you compare across epochs are apples
    # to apples.
    train_dataset = IndianaChestXRayDataset(
        df[train_mask], label_vectors[train_mask], label_names, train=True
    )
    val_dataset = IndianaChestXRayDataset(
        df[val_mask], label_vectors[val_mask], label_names, train=False
    )
    test_dataset = IndianaChestXRayDataset(
        df[test_mask], label_vectors[test_mask], label_names, train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, label_names


# ─── TEST / DEMO ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Indiana Chest X-Ray Dataset - Demo")
    print("=" * 60)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, label_names = get_dataloaders(
        batch_size=4, num_workers=0
    )
    
    print(f"\nLabel names ({len(label_names)}): {label_names}")
    
    # Load one batch and inspect
    images, labels = next(iter(train_loader))
    
    print(f"\n=== Sample Batch ===")
    print(f"Image tensor shape: {images.shape}")   # (batch, 3, 224, 224)
    print(f"Image dtype:        {images.dtype}")     # float32
    print(f"Image range:        [{images.min():.2f}, {images.max():.2f}]")  # ImageNet normalized
    print(f"Label tensor shape: {labels.shape}")     # (batch, n_labels)
    print(f"Label dtype:        {labels.dtype}")     # float32
    
    # Show which labels are active for the first sample
    print(f"\nFirst sample labels:")
    for i, name in enumerate(label_names):
        if labels[0, i] == 1.0:
            print(f"  [x] {name}")