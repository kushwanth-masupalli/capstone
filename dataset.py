"""
Indiana Chest X-Ray Dataset
============================
PyTorch Dataset class for loading and preparing chest X-ray images.

Steps:
1. Merge indiana_reports.csv + indiana_projections.csv on 'uid'
2. Filter to Frontal (PA) images only
3. Parse the 'MeSH' column into multi-label binary encoding (with anatomy-qualifier recovery)
4. Load preprocessed images from data/images/preprocessed/
5. Apply normalization on-the-fly (ImageNet or TorchXRayVision)

Backbone options:
    backbone="imagenet" (default): 3-channel RGB, ImageNet normalization
    backbone="xrv": 1-channel grayscale, XRV normalization ([-1024, 1024])

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)
    # For XRV backbone:
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, backbone="xrv")
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
REPORTS_CSV = DATA_DIR / "indiana_reports.csv"
PROJECTIONS_CSV = DATA_DIR / "indiana_projections.csv"
IMAGES_DIR = DATA_DIR / "images" / "preprocessed"

# Multi-label: use top N most common individual problems (pathology only)
# "normal" is handled implicitly, not as a prediction target
TOP_N_PROBLEMS = 7
MIN_SAMPLES = 10  # Minimum samples per problem to include

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Random seed for reproducibility
RANDOM_STATE = 42

# Default backbone (set to "imagenet" or "xrv")
DEFAULT_BACKBONE = "imagenet"


# ─── TWO-HEAD TAXONOMY ──────────────────────────────────────────────────────
# Labels split into pathology (disease findings → used for hallucination
# detection) and anatomy/device (structural/equipment → auxiliary only).
#
# IMPORTANT: this list must match whatever build_label_encoder() actually
# produces. It was regenerated after the anatomy-qualifier recovery fix (see
# recover_mesh_term() below) changed which labels exist — e.g. "Spine" used
# to always mean a bare, meaningless anatomy mention; now most "Spine"
# mentions recover into "Degenerative Spine Disease" and only report entries
# with no recoverable finding fall back to bare "Spine". If you rerun label
# extraction and get a different top-15, update these two lists to match —
# build_label_encoder() will raise if they don't, rather than silently
# mislabeling columns.
#
# These lists define the *target* ordering: pathology labels come first,
# anatomy labels second.  build_label_encoder() still returns a single
# flat vector — the reordering is applied by reorder_labels() after the
# encoder runs, so dataset.py stays backward-compatible.

# "normal" is NOT a pathology — it's the absence of any finding.
# We handle it implicitly: if no pathology predicted → "normal".
# This removes the dominant class that biases the model.
PATHOLOGY_LABELS = [
    "Granuloma",
    "Opacity",
    "Degenerative Spine Disease",
    "Cardiomegaly",
    "Pulmonary Atelectasis",
    "Pulmonary Hypoinflation",
    "Pulmonary Hyperinflation",
]

ANATOMY_LABELS = []  # Disable anatomy head initially

# TOP_N_PROBLEMS now refers to pathology labels only (was 8, now 7)
assert len(PATHOLOGY_LABELS) + len(ANATOMY_LABELS) == 7, (
    f"Taxonomy size mismatch: {len(PATHOLOGY_LABELS)} pathology + "
    f"{len(ANATOMY_LABELS)} anatomy != 7"
)


def label_group(name: str) -> str:
    """Return 'pathology' or 'anatomy' for a given label name."""
    if name in PATHOLOGY_LABELS:
        return "pathology"
    if name in ANATOMY_LABELS:
        return "anatomy"
    raise ValueError(f"Unknown label: {name}")


def reorder_labels(label_names: list, label_vectors: np.ndarray):
    """
    Reorder labels so pathology columns come first, anatomy second.

    Returns:
        new_names: reordered list of label names
        new_vectors: np.ndarray with columns reordered to match
    """
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    ordered = PATHOLOGY_LABELS + ANATOMY_LABELS
    new_vectors = np.column_stack([label_vectors[:, name_to_idx[n]] for n in ordered])
    return ordered, new_vectors


# ─── DATA PREPARATION ────────────────────────────────────────────────────────

def load_and_merge_data():
    """
    Load both CSVs, merge on 'uid', filter to Frontal images only.
    
    Returns:
        pd.DataFrame with columns: uid, filename, projection, Problems, MeSH, label_vector
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

    # Recovered anatomy+finding compounds (see recover_mesh_term). Thoracic
    # Vertebrae and Spine degenerative changes are the same clinical concept
    # under two different MeSH headings, so they merge. Diaphragm elevation
    # vs. flattening are clinically distinct (elevation suggests atelectasis/
    # paralysis, flattening suggests hyperinflation/COPD) so they stay separate
    # if they ever surface as their own compound terms.
    "Thoracic Vertebrae - degenerative": "Degenerative Spine Disease",
    "Spine - degenerative": "Degenerative Spine Disease",
    "Cervical Vertebrae - degenerative": "Degenerative Spine Disease",
    "Lung - hyperdistention": "Pulmonary Hyperinflation",
    "Lung - hypoinflation": "Pulmonary Hypoinflation",
    "Aorta - tortuous": "Tortuous Aorta",
    "Aorta - enlarged": "Aortic Enlargement",
    "Mediastinum - prominent": "Prominent Mediastinum",
}

# Noise labels to remove entirely
NOISE_LABELS = {
    "No Indexing",
    "Technical Quality of Image Unsatisfactory",
    "Tube, Inserted",
    "Implanted Medical Device",
    "normal",  # "normal" is not a pathology target — handled implicitly
}


# ─── ANATOMY-QUALIFIER RECOVERY ─────────────────────────────────────────────
# The CSV's 'Problems' column is PRE-STRIPPED of qualifiers (e.g. the raw MeSH
# entry "Lung/hyperdistention" becomes just "Lung" in 'Problems'). For terms
# where the finding itself is a real diagnosis noun (Cicatrix, Opacity,
# Cardiomegaly, Granuloma, ...), that first token is already correct and
# nothing is lost. But for a specific set of terms, the MeSH convention orders
# it as "anatomy site / descriptor" instead — e.g. "Lung/hyperdistention",
# "Aorta/tortuous", "Spine/degenerative", "Diaphragm/right/elevated" — and
# taking only the first token there throws away the actual finding, leaving a
# meaningless bare anatomy label ("Lung", "Aorta", "Spine", "Diaphragm").
# Verified against this dataset: the large majority of every such mention has
# a real, recoverable finding word after the anatomy term — this is not a
# rare edge case, it affects hundreds of images per anatomy term. What's left
# after recovery (bare "Aorta"/"Diaphragm"/"Spine" with only a location word
# and no finding) is genuinely uninformative and is what ANATOMY_LABELS above
# now represents.
#
# We only special-case this specific, closed set of anatomy-first base terms.
# Every other term keeps its original first-token-only behavior, since that's
# already correct for diagnosis-first terms.

ANATOMY_BASE_TERMS = {
    "Lung", "Spine", "Aorta", "Diaphragm", "Thoracic Vertebrae",
    "Mediastinum", "Heart", "Trachea", "Ribs", "Cervical Vertebrae",
}

# Pure location/laterality/multiplicity words — never the actual finding,
# skip past these when looking for the real descriptor.
LOCATION_WORDS = {
    "right", "left", "bilateral", "anterior", "posterior", "upper", "lower",
    "mid", "base", "apex", "hilum", "hilar", "multiple", "scattered",
    "blood vessels", "lymph nodes", "retrocardiac", "upper lobe",
    "lower lobe", "middle lobe", "cardiophrenic angle",
}


def recover_mesh_term(raw_term):
    """
    Given one raw MeSH entry (e.g. "Lung/hyperdistention/mild"), return the
    label that should represent it.

    - If the base (first) token is already a real diagnosis noun, return it
      unchanged — this matches the dataset's original (correct) behavior.
    - If the base token is a pure anatomy word, scan the qualifiers for the
      first one that isn't a location/laterality word, and combine it with
      the anatomy term (e.g. "Lung/hyperdistention/mild" -> "Lung -
      hyperdistention"), recovering the finding that would otherwise be
      discarded.
    - If no recoverable finding qualifier exists, fall back to the bare
      anatomy term — this is the genuinely uninformative case ANATOMY_LABELS
      represents.
    """
    parts = [p.strip() for p in raw_term.split("/")]
    base = parts[0]
    if base not in ANATOMY_BASE_TERMS:
        return base
    for q in parts[1:]:
        if q.lower() not in LOCATION_WORDS:
            return f"{base} - {q}"
    return base


def clean_problems(mesh_str):
    """
    Clean a raw MeSH string into standardized label names.
    
    Steps:
    1. Split by semicolon
    2. Remove noise labels
    3. Recover findings hidden behind anatomy-first terms (see
       recover_mesh_term above) — this is the key difference from using the
       pre-stripped 'Problems' column directly.
    4. Normalize names using PROBLEM_NAME_MAP
    5. Deduplicate
    
    Args:
        mesh_str: raw string from the 'MeSH' column, e.g.
            "Calcified Granuloma/lung/upper lobe/right;Density/cardiophrenic angle/left"
        
    Returns:
        list of clean, deduplicated label names
    """
    if pd.isna(mesh_str):
        return []
    
    # Split by semicolon, strip whitespace
    raw_labels = [p.strip() for p in str(mesh_str).split(";") if p.strip()]
    
    # Remove noise and normalize
    clean_labels = []
    for label in raw_labels:
        # Skip noise (check the base term before recovery)
        base_term = label.split("/")[0].strip()
        if base_term in NOISE_LABELS or label in NOISE_LABELS:
            continue
        
        # Recover the real finding for anatomy-first compound terms
        recovered = recover_mesh_term(label)
        
        # Map to normalized name
        normalized = PROBLEM_NAME_MAP.get(recovered, None)
        
        # If not in map, try partial matching on the base term
        if normalized is None:
            for key, value in PROBLEM_NAME_MAP.items():
                if key in recovered:
                    normalized = value
                    break
        
        # Keep as-is if no mapping found (but still valid)
        if normalized is None:
            normalized = recovered
        
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
    Parse 'MeSH' column into multi-label binary encoding.
    
    The MeSH column contains semicolon-separated MeSH entries with qualifiers like:
        "Calcified Granuloma/lung/upper lobe/right;Density/cardiophrenic angle/left"
    
    Steps:
    1. Clean and normalize problem names (with anatomy-qualifier recovery)
    2. Remove noise labels
    3. Deduplicate within each entry
    4. Count frequency and keep top N
    
    Args:
        df: DataFrame with 'MeSH' column
        
    Returns:
        label_names: list of problem names (e.g., ['normal', 'Cardiomegaly', ...])
        label_vectors: np.ndarray of shape (n_samples, n_labels) with 0/1 values
    """
    # Clean and count problems
    problem_counter = Counter()
    all_problems = []
    
    for mesh_str in df["MeSH"]:
        clean_labels = clean_problems(mesh_str)
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
    
    # Reorder: pathology labels first, anatomy second
    label_names, label_vectors = reorder_labels(label_names, label_vectors)
    
    n_path = len(PATHOLOGY_LABELS)
    n_anat = len(ANATOMY_LABELS)
    print(f"\nLabel groups (after reorder):")
    print(f"  Pathology ({n_path}): {label_names[:n_path]}")
    print(f"  Anatomy   ({n_anat}): {label_names[n_path:]}")
    
    return label_names, label_vectors


# ─── XRV PREPROCESSING PIPELINE ──────────────────────────────────────────────
# TorchXRayVision expects single-channel grayscale input with its own
# normalize/resize/crop pipeline. This custom class wraps the xrv transforms
# so they integrate with torchvision-style augmentation.

class XRVPreprocessor:
    """
    Custom transform that handles the full XRV preprocessing pipeline:
        1. Load as grayscale (L mode)
        2. Apply augmentation (if training)
        3. Convert to numpy for xrv processing
        4. Apply xrv.datasets.normalize() (maps to [-1024, 1024])
        5. Apply XRayResizer(224)
        6. Apply XRayCenterCrop()
        7. Convert to tensor with shape (1, 224, 224)
    """

    def __init__(self, train=False):
        import torchxrayvision as xrv
        self.train = train
        self.resizer = xrv.datasets.XRayResizer(224)
        self.cropper = xrv.datasets.XRayCenterCrop()
        # Augmentation: more aggressive to match imagenet pipeline
        if train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.85, 1.15),
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.augment = None

    def __call__(self, img):
        """
        Args:
            img: PIL Image (RGB or L)
        Returns:
            torch.Tensor of shape (1, 224, 224), float32
        """
        import torchxrayvision as xrv
        import numpy as np

        # Convert to grayscale
        img = img.convert("L")

        # Apply augmentation on PIL image (before converting to numpy)
        if self.augment is not None:
            img = self.augment(img)

        # Convert to numpy: (H, W) -> (1, H, W) with range [0, 255]
        img_np = np.array(img).astype(np.float32)
        img_np = img_np[np.newaxis, ...]  # (1, H, W)

        # XRV normalize expects 0-255 input
        img_np = xrv.datasets.normalize(img_np, maxval=255.0)  # maps to [-1024, 1024]

        # Resize and center crop
        img_np = self.resizer(img_np)
        img_np = self.cropper(img_np)

        # Convert to tensor: (1, H, W) -> (1, 1, 224, 224)
        return torch.from_numpy(img_np).float()


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

    More aggressive augmentation to improve generalization with a small
    dataset. RandomResizedCrop can help avoid cropping out pathology by
    varying the crop area. ColorJitter simulates different exposure
    settings common in clinical X-rays.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=15,             # slightly larger rotation
            translate=(0.1, 0.1),    # larger shift
            scale=(0.85, 1.15),     # larger zoom range
        ),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── PYTORCH DATASET ─────────────────────────────────────────────────────────

class IndianaChestXRayDataset(Dataset):
    """
    PyTorch Dataset for Indiana Chest X-Ray images.

    Loads preprocessed images (224x224, CLAHE-enhanced) and applies
    normalization on-the-fly.

    Supports two backbone modes:
        backbone="imagenet": RGB input, ImageNet normalization, shape (3, 224, 224)
        backbone="xrv": Grayscale input, XRV normalization, shape (1, 224, 224)
    """

    def __init__(self, df, label_vectors, label_names, transform=None,
                 train=False, backbone="imagenet"):
        """
        Args:
            df: DataFrame with 'filename' column
            label_vectors: np.ndarray of shape (n_samples, n_labels)
            label_names: list of problem names
            transform: torchvision transforms to apply. If None, falls back to
                backbone-specific defaults.
            train: whether this split should get augmentation when transform
                is not explicitly given.
            backbone: "imagenet" (default) or "xrv" for TorchXRayVision.
        """
        self.df = df.reset_index(drop=True)
        self.label_vectors = label_vectors
        self.label_names = label_names
        self.filenames = df["filename"].values
        self.backbone = backbone

        if transform is None:
            if backbone == "xrv":
                self.transform = XRVPreprocessor(train=train)
            else:
                self.transform = get_train_transform() if train else get_eval_transform()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Load one image and its multi-label vector.

        Returns:
            image: torch.Tensor, shape (3, 224, 224) for imagenet or (1, 224, 224) for xrv
            labels: torch.Tensor of shape (n_labels,), float32 (0/1)
        """
        filename = self.filenames[idx]
        img_path = IMAGES_DIR / filename

        # Load image as RGB (XRVPreprocessor handles grayscale conversion internally)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        image = self.transform(image)

        # Get label vector
        labels = torch.tensor(self.label_vectors[idx], dtype=torch.float32)

        return image, labels


def get_balanced_sampler(label_vectors):
    """
    Create a WeightedRandomSampler that oversamples rare classes.

    Each sample's weight = sum of per-class inverse-frequency weights for
    the labels that are active in that sample.  This means images with
    rare labels get drawn more often, preventing the model from ignoring
    minority pathologies.

    Args:
        label_vectors: np.ndarray of shape (n_samples, n_labels), 0/1

    Returns:
        WeightedRandomSampler with replacement=True
    """
    n_samples, n_labels = label_vectors.shape
    pos_counts = label_vectors.sum(axis=0)  # (n_labels,)
    # Inverse frequency per class: rare classes get higher weight
    class_weights = 1.0 / (pos_counts + 1.0)  # +1 to avoid div-by-zero
    # Per-sample weight = sum of class weights for active labels
    sample_weights = (label_vectors * class_weights).sum(axis=1)
    # Ensure no sample has zero weight (images with no labels get min weight)
    sample_weights = np.maximum(sample_weights, sample_weights[sample_weights > 0].min())

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_samples,
        replacement=True,
    )
    print(f"\nBalanced sampling enabled: {n_samples} samples, "
          f"rarest active-class weight={sample_weights.min():.4f}, "
          f"most common weight={sample_weights.max():.4f}")
    return sampler


# ─── DATA LOADER FACTORY ─────────────────────────────────────────────────────
def get_splits(test_size=0.15, val_size=0.15):
    """
    Load + merge data, build labels, and split by patient (uid).

    Returns:
        df: full merged DataFrame
        label_names: list of pathology label names
        label_vectors: (n_samples, n_labels) 0/1 matrix aligned to df rows
        train_mask, val_mask, test_mask: boolean Series aligned to df rows
    """
    df = load_and_merge_data()
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

    return df, label_names, label_vectors, train_mask, val_mask, test_mask


def get_dataloaders(batch_size=16, num_workers=0, test_size=0.15, val_size=0.15,
                    backbone="imagenet", balanced_sampling=False):
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
        backbone: "imagenet" (default) or "xrv" for TorchXRayVision
        balanced_sampling: If True, use WeightedRandomSampler for training

    Returns:
        train_loader, val_loader, test_loader, label_names
    """
    df, label_names, label_vectors, train_mask, val_mask, test_mask = get_splits(
        test_size=test_size, val_size=val_size
    )

    print(f"Backbone: {backbone}")

    # Create datasets -- only the training split gets augmentation; val/test
    # stay deterministic so the metrics you compare across epochs are apples
    # to apples.
    train_dataset = IndianaChestXRayDataset(
        df[train_mask], label_vectors[train_mask], label_names,
        train=True, backbone=backbone
    )
    val_dataset = IndianaChestXRayDataset(
        df[val_mask], label_vectors[val_mask], label_names,
        train=False, backbone=backbone
    )
    test_dataset = IndianaChestXRayDataset(
        df[test_mask], label_vectors[test_mask], label_names,
        train=False, backbone=backbone
    )

    # Create data loaders
    if balanced_sampling:
        sampler = get_balanced_sampler(train_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True
        )
    else:
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

    # Test both backbone pipelines
    for backbone in ["imagenet", "xrv"]:
        print(f"\n{'=' * 40}")
        print(f"Testing backbone: {backbone}")
        print(f"{'=' * 40}")

        train_loader, val_loader, test_loader, label_names = get_dataloaders(
            batch_size=4, num_workers=0, backbone=backbone
        )

        print(f"\nLabel names ({len(label_names)}): {label_names}")

        # Load one batch and inspect
        images, labels = next(iter(train_loader))

        print(f"\n--- Sample Batch ---")
        print(f"Image tensor shape: {images.shape}")
        print(f"Image dtype:        {images.dtype}")
        print(f"Image range:        [{images.min():.2f}, {images.max():.2f}]")
        print(f"Label tensor shape: {labels.shape}")
        print(f"Label dtype:        {labels.dtype}")

        # Show which labels are active for the first sample
        print(f"\nFirst sample labels:")
        for i, name in enumerate(label_names):
            if labels[0, i] == 1.0:
                print(f"  [x] {name}")
