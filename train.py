"""
DenseNet121 Chest X-Ray Training Script
=========================================
Trains a DenseNet121 model on Indiana Chest X-Ray dataset for multi-label
classification of chest pathology.

Usage:
    python train.py                              # Default (ImageNet backbone)
    python train.py --backbone xrv               # TorchXRayVision backbone
    python train.py --backbone xrv --xrv_checkpoint densenet121-res224-nih
    python train.py --epochs 30                  # Custom epochs
    python train.py --batch_size 32              # Custom batch size

Pseudo-label expansion (semi-supervised self-training, see PSEUDO_LABELING.md):
    python train.py --backbone xrv --epochs 80 --balanced_sampling --focal_loss \
        --save_prefix pseudo_ --pseudo_csv data/pseudo_labels.csv
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

# Optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except Exception:
    HAS_TENSORBOARD = False

from dataset import get_dataloaders


# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Training hyperparameters
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3       # Higher LR for heads (was 3e-4)
DEFAULT_BACKBONE_LR = 1e-5         # Much lower LR for backbone (was 5e-5)
DEFAULT_WEIGHT_DECAY = 1e-4        # Stronger weight decay (was 1e-5)
DEFAULT_PATIENCE = 20              # More patience (was 15)
DEFAULT_MAX_POS_WEIGHT = 50.0      # Much higher cap for rare classes (was 10.0)
DEFAULT_FREEZE_EPOCHS = 5          # Freeze backbone longer initially (was 2)

# XRV-specific defaults (more aggressive fine-tuning since already CXR-pretrained)
XRV_BACKBONE_LR = 5e-5             # Higher LR for CXR-pretrained backbone
XRV_FREEZE_EPOCHS = 3              # Freeze for a few epochs even for XRV
XRV_DROPOUT = 0.5                  # Stronger dropout for XRV heads (was 0.3)

# ResNet50-specific defaults
RESNET50_FEATURE_DIM = 2048

# Focal loss defaults
DEFAULT_FOCAL_ALPHA = 0.25
DEFAULT_FOCAL_GAMMA = 2.0

# Label smoothing (0.0 = disabled) — enables soft labels to prevent overconfidence
DEFAULT_LABEL_SMOOTHING = 0.05

# Model
NUM_LABELS = 7  # Top 7 pathology problems (normal handled implicitly)
PRETRAINED = True  # Use ImageNet pretrained weights

# Checkpointing
CHECKPOINT_DIR = Path("checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── MODEL ───────────────────────────────────────────────────────────────────

from dataset import PATHOLOGY_LABELS, ANATOMY_LABELS, label_group

N_PATHOLOGY = len(PATHOLOGY_LABELS)  # 10
N_ANATOMY = len(ANATOMY_LABELS)      # 5


class TwoHeadClassifier(nn.Module):
    """
    Two-head classifier: any backbone's feature extractor + two linear heads.

    pathology_head: outputs logits for disease-finding labels (used for
        hallucination detection in Phase 8).
    anatomy_head:   outputs logits for structure/device labels (auxiliary
        task only — improves shared backbone features).

    forward() concatenates both heads in a fixed order: pathology columns
    first, anatomy columns second.  This keeps the rest of the pipeline
    (validate, threshold tuning, per-class metrics) working unchanged.
    """

    def __init__(self, features_module, feature_dim=1024,
                 n_pathology=N_PATHOLOGY, n_anatomy=N_ANATOMY,
                 dropout=0.5):
        super().__init__()
        self.features = features_module
        self.feature_dim = feature_dim
        self.n_pathology = n_pathology
        self.n_anatomy = n_anatomy

        self.pathology_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, n_pathology),
        )
        # Only create anatomy head if there are anatomy labels
        if n_anatomy > 0:
            self.anatomy_head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, n_anatomy),
            )
        else:
            self.anatomy_head = None

    def forward(self, x):
        feat = self.features(x)
        feat = torch.nn.functional.relu(feat)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)              # (B, feature_dim)

        path_out = self.pathology_head(feat)       # (B, n_pathology)
        if self.anatomy_head is not None:
            anat_out = self.anatomy_head(feat)      # (B, n_anatomy)
            return torch.cat([path_out, anat_out], dim=1)
        return path_out  # (B, n_pathology)


def create_model(num_labels=NUM_LABELS, pretrained=PRETRAINED,
                 backbone="imagenet", xrv_checkpoint="densenet121-res224-chex"):
    """
    Create two-head classifier with various backbone options.

    Supports three backbones:
        backbone="imagenet": torchvision DenseNet121 pretrained on ImageNet
        backbone="xrv": TorchXRayVision DenseNet pretrained on CXR datasets
        backbone="resnet50": torchvision ResNet50 pretrained on ImageNet

    Args:
        num_labels: Total number of output labels (pathology + anatomy)
        pretrained: Use ImageNet pretrained weights (only for imagenet/resnet50)
        backbone: "imagenet", "xrv", or "resnet50"
        xrv_checkpoint: XRV weights to load (only for xrv backbone)

    Returns:
        model: PyTorch model with two-head classifier
    """
    assert num_labels == N_PATHOLOGY + N_ANATOMY, (
        f"num_labels ({num_labels}) != N_PATHOLOGY ({N_PATHOLOGY}) + "
        f"N_ANATOMY ({N_ANATOMY}) = {N_PATHOLOGY + N_ANATOMY}"
    )

    if backbone == "xrv":
        import torchxrayvision as xrv
        print(f"Loading XRV DenseNet: {xrv_checkpoint}")
        xrv_model = xrv.models.DenseNet(weights=xrv_checkpoint)
        model = TwoHeadClassifier(xrv_model.features, feature_dim=1024, dropout=XRV_DROPOUT).to(DEVICE)
        print(f"Model: DenseNet121 Two-Head (XRV backbone)")
        print(f"  Checkpoint: {xrv_checkpoint}")
        print(f"  Dropout: {XRV_DROPOUT}")
        # Verify feature dimensions match expectations
        with torch.no_grad():
            _dummy = torch.randn(1, 1, 224, 224, device=DEVICE)
            _feat = model.features(_dummy)
            _feat = torch.nn.functional.relu(_feat)
            _feat = torch.nn.functional.adaptive_avg_pool2d(_feat, (1, 1))
            _feat = torch.flatten(_feat, 1)
            print(f"  Feature verification: features->({list(model.features(_dummy).shape)}) -> head({_feat.shape[1]})")
            del _dummy, _feat
    elif backbone == "resnet50":
        from torchvision import models
        print("Loading ResNet50 backbone")
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)
        # Remove avgpool and fc, keep everything up to layer4
        resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        model = TwoHeadClassifier(resnet_features, feature_dim=RESNET50_FEATURE_DIM).to(DEVICE)
        print(f"Model: ResNet50 Two-Head (ImageNet backbone)")
        # Verify feature dimensions
        with torch.no_grad():
            _dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
            _feat = model.features(_dummy)
            _feat = torch.nn.functional.relu(_feat)
            _feat = torch.nn.functional.adaptive_avg_pool2d(_feat, (1, 1))
            _feat = torch.flatten(_feat, 1)
            print(f"  Feature verification: features->({list(model.features(_dummy).shape)}) -> head({_feat.shape[1]})")
            del _dummy, _feat
    else:  # imagenet
        from torchvision import models
        if pretrained:
            densenet = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1
            )
        else:
            densenet = models.densenet121(weights=None)
        model = TwoHeadClassifier(densenet.features, feature_dim=1024)
        print(f"Model: DenseNet121 Two-Head (ImageNet backbone)")

    feat_dim = model.feature_dim
    print(f"  Features: {feat_dim}")
    print(f"  Pathology head: {N_PATHOLOGY} labels  {PATHOLOGY_LABELS}")
    print(f"  Anatomy head:   {N_ANATOMY} labels  {ANATOMY_LABELS}")
    print(f"  Output: {num_labels} labels (pathology-first, anatomy-second)")
    print(f"  Device: {DEVICE}")
    return model.to(DEVICE)


def set_backbone_trainable(model, trainable):
    """
    Freeze or unfreeze the shared feature extractor only.

    Both heads always train — they're small and randomly initialized.
    """
    for name, param in model.named_parameters():
        if name.startswith("features"):
            param.requires_grad = trainable


# ─── LOSS FUNCTION ───────────────────────────────────────────────────────────


def focal_bce_loss(logits, targets, pos_weight, alpha=0.25, gamma=2.0):
    """
    Focal loss variant for multi-label binary classification.

    Down-weights easy negatives so rare positive classes get more gradient
    signal.  Equivalent to multiplying BCE by (1 - p_t)^gamma where
    p_t = predicted probability of the true class.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)  # probability of true class
    focal_weight = (1 - pt) ** gamma
    # Apply alpha: upweight positives, downweight negatives
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * focal_weight * bce).mean()


def compute_pos_weights(label_vectors_train, max_pos_weight=DEFAULT_MAX_POS_WEIGHT):
    """
    Compute capped inverse-frequency pos_weight for each label column.
    """
    pos_counts = label_vectors_train.sum(axis=0)
    total = label_vectors_train.shape[0]
    weights = total / (2 * pos_counts + 1e-6)
    weights = np.clip(weights, a_min=None, a_max=max_pos_weight)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def create_two_head_loss(label_vectors_train, max_pos_weight=DEFAULT_MAX_POS_WEIGHT,
                         anatomy_loss_weight=0.3, use_focal=False,
                         focal_alpha=DEFAULT_FOCAL_ALPHA, focal_gamma=DEFAULT_FOCAL_GAMMA,
                         label_smoothing=0.0):
    """
    Create a two-head weighted loss.

    Args:
        use_focal: use focal loss instead of standard BCE
        focal_alpha: focal loss alpha parameter
        focal_gamma: focal loss gamma parameter (higher = more focus on hard examples)
        label_smoothing: soft-label smoothing factor (0.0 = disabled)

    Returns:
        loss_fn(images, labels, model) -> (total_loss, path_loss, anat_loss)
        all_weights: full weight tensor for printing
    """
    all_weights = compute_pos_weights(label_vectors_train, max_pos_weight)
    path_weights = all_weights[:N_PATHOLOGY]
    anat_weights = all_weights[N_PATHOLOGY:] if N_ANATOMY > 0 else None

    if use_focal:
        print(f"\nUsing Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        def _focal(logits, targets, pw):
            return focal_bce_loss(logits, targets, pw, alpha=focal_alpha, gamma=focal_gamma)
        path_criterion = _focal
        anat_criterion = _focal
    else:
        path_criterion = nn.BCEWithLogitsLoss(pos_weight=path_weights)
        anat_criterion = nn.BCEWithLogitsLoss(pos_weight=anat_weights) if N_ANATOMY > 0 else None

    print(f"\nClass weights (pathology head):")
    for i, name in enumerate(PATHOLOGY_LABELS):
        print(f"  {name:25s} pos={int(label_vectors_train[:, i].sum()):5d}  weight={path_weights[i]:.2f}")
    if N_ANATOMY > 0:
        print(f"\nClass weights (anatomy head, weight={anatomy_loss_weight}):")
        for i, name in enumerate(ANATOMY_LABELS):
            j = N_PATHOLOGY + i
            print(f"  {name:25s} pos={int(label_vectors_train[:, j].sum()):5d}  weight={anat_weights[i]:.2f}")

    def loss_fn(images, labels, model):
        outputs = model(images)

        # Apply label smoothing: move labels toward 0.5 by (1 - ls) * label + ls * 0.5
        if label_smoothing > 0.0:
            smooth_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            smooth_labels = labels

        if N_ANATOMY > 0 and anat_criterion is not None:
            if use_focal:
                path_loss = path_criterion(
                    outputs[:, :N_PATHOLOGY], smooth_labels[:, :N_PATHOLOGY], path_weights
                )
                anat_loss = anat_criterion(
                    outputs[:, N_PATHOLOGY:], smooth_labels[:, N_PATHOLOGY:], anat_weights
                )
            else:
                path_loss = path_criterion(outputs[:, :N_PATHOLOGY], smooth_labels[:, :N_PATHOLOGY])
                anat_loss = anat_criterion(outputs[:, N_PATHOLOGY:], smooth_labels[:, N_PATHOLOGY:])
            total = path_loss + anatomy_loss_weight * anat_loss
        else:
            # Pathology only — no anatomy head
            if use_focal:
                path_loss = path_criterion(outputs, smooth_labels, path_weights)
            else:
                path_loss = path_criterion(outputs, smooth_labels)
            anat_loss = torch.tensor(0.0, device=images.device)
            total = path_loss

        return total, path_loss, anat_loss

    return loss_fn, all_weights


# ─── CUTMIX / MIXUP ────────────────────────────────────────────────────────

class CutMix:
    """CutMix augmentation for multi-label classification.

    Cuts a random rectangular region from one image and pastes it onto
    another, mixing labels proportionally to the area ratio.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, images, labels):
        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        rand_idx = torch.randperm(batch_size, device=images.device)

        # Random bounding box
        _, _, H, W = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)

        images_mixed = images.clone()
        images_mixed[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]

        # Adjust lambda by actual area ratio
        lam_actual = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
        labels_mixed = lam_actual * labels + (1 - lam_actual) * labels[rand_idx]

        return images_mixed, labels_mixed


class MixUp:
    """MixUp augmentation — interpolates images and labels."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, images, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        rand_idx = torch.randperm(images.size(0), device=images.device)
        images_mixed = lam * images + (1 - lam) * images[rand_idx]
        labels_mixed = lam * labels + (1 - lam) * labels[rand_idx]
        return images_mixed, labels_mixed


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch,
                     mix_fn=None):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_path_loss = 0.0
    total_anat_loss = 0.0
    all_preds = []
    all_labels = []  # Store ORIGINAL hard labels for metrics

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Keep original hard labels for metric computation
        original_labels = labels.clone()

        # Apply CutMix / MixUp if enabled
        if mix_fn is not None:
            images, labels = mix_fn(images, labels)

        # Forward + two-head loss
        loss, path_loss, anat_loss = loss_fn(images, labels, model)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Track metrics
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_path_loss += path_loss.item() * bs
        total_anat_loss += anat_loss.item() * bs

        with torch.no_grad():
            outputs = model(images)
            preds = (outputs > 0.0).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(original_labels.cpu().numpy())  # Use hard labels for metrics

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} (path: {path_loss.item():.4f} anat: {anat_loss.item():.4f})")

    # Calculate epoch metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    n = len(train_loader.dataset)

    avg_loss = total_loss / n
    avg_path = total_path_loss / n
    avg_anat = total_anat_loss / n
    f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)

    return avg_loss, avg_path, avg_anat, f1, precision, recall


def validate(model, val_loader, loss_fn=None, thresholds=None):
    """
    Validate the model.

    Args:
        loss_fn: two-head loss function (from create_two_head_loss). If None,
            uses plain BCE for backward compat.
        thresholds: optional np.ndarray of shape (n_labels,) with a
            per-class decision threshold. Defaults to 0.5 for every class.

    Returns a dict of metrics plus raw probs/labels.
    """
    model.eval()
    
    total_loss = 0.0
    total_path_loss = 0.0
    total_anat_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            if loss_fn is not None:
                # Compute validation loss directly from outputs (no extra forward pass)
                # Use plain unweighted BCE so val loss tracks clean signal
                path_loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs[:, :N_PATHOLOGY], labels[:, :N_PATHOLOGY])
                total_path_loss += path_loss.item() * images.size(0)
                total_loss += path_loss.item() * images.size(0)
                if N_ANATOMY > 0:
                    anat_loss = nn.functional.binary_cross_entropy_with_logits(
                        outputs[:, N_PATHOLOGY:], labels[:, N_PATHOLOGY:])
                    total_anat_loss += anat_loss.item() * images.size(0)
                    total_loss += anat_loss.item() * images.size(0)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels)
                total_loss += loss.item() * images.size(0)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    n = len(val_loader.dataset)

    if thresholds is None:
        thresholds = np.full(all_probs.shape[1], 0.5, dtype=np.float32)
    all_preds = (all_probs > thresholds[np.newaxis, :]).astype(np.float32)
    
    avg_loss = total_loss / n
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, average="micro")
    except ValueError:
        auc = 0.0
    
    result = {
        "loss": avg_loss,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "probs": all_probs,
        "labels": all_labels,
    }

    # Add split-group metrics if two-head loss was provided
    if loss_fn is not None and total_path_loss > 0:
        result["path_loss"] = total_path_loss / n
        result["anat_loss"] = total_anat_loss / n
        # Pathology-only metrics (columns 0..N_PATHOLOGY-1)
        p_probs = all_probs[:, :N_PATHOLOGY]
        p_labels = all_labels[:, :N_PATHOLOGY]
        p_preds = all_preds[:, :N_PATHOLOGY]
        result["path_f1_micro"] = f1_score(p_labels, p_preds, average="micro", zero_division=0)
        result["path_f1_macro"] = f1_score(p_labels, p_preds, average="macro", zero_division=0)
        try:
            result["path_auc"] = roc_auc_score(p_labels, p_probs, average="micro")
        except ValueError:
            result["path_auc"] = 0.0
        # Anatomy-only metrics (columns N_PATHOLOGY..end)
        if N_ANATOMY > 0:
            a_probs = all_probs[:, N_PATHOLOGY:]
            a_labels = all_labels[:, N_PATHOLOGY:]
            a_preds = all_preds[:, N_PATHOLOGY:]
            result["anat_f1_micro"] = f1_score(a_labels, a_preds, average="micro", zero_division=0)
            try:
                result["anat_auc"] = roc_auc_score(a_labels, a_probs, average="micro")
            except ValueError:
                result["anat_auc"] = 0.0
        else:
            result["anat_f1_micro"] = 0.0
            result["anat_auc"] = 0.0

    return result


def find_optimal_thresholds(probs, labels, label_names):
    """
    Pick, per class, the decision threshold (0.05 .. 0.90) that maximizes F1
    on the given probabilities/labels.

    A single global 0.5 threshold is a bad fit here: pos_weight skews each
    class's probability distribution differently, so the threshold that
    balances precision/recall for 'normal' (1,387 positives) is not the
    right threshold for 'Density' (48 positives). This is tuned on the
    validation set only, never on test.
    """
    n_labels = probs.shape[1]
    thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.arange(0.05, 0.95, 0.05)

    print("\n--- Tuning per-class decision thresholds on validation set ---")
    for i, name in enumerate(label_names):
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            preds = (probs[:, i] > t).astype(np.float32)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
        print(f"  {name:25s} threshold={best_t:.2f}  val F1={best_f1:.4f}")

    return thresholds


def print_per_class_metrics(probs, labels, thresholds, label_names, title):
    """Print F1/precision/recall for every class individually."""
    preds = (probs > thresholds[np.newaxis, :]).astype(np.float32)
    print(f"\n--- {title}: Per-Class Metrics ---")
    print(f"{'Label':25s} {'F1':>6} {'Precision':>10} {'Recall':>8} {'Support':>8}")
    for i, name in enumerate(label_names):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        prec = precision_score(labels[:, i], preds[:, i], zero_division=0)
        rec = recall_score(labels[:, i], preds[:, i], zero_division=0)
        support = int(labels[:, i].sum())
        print(f"{name:25s} {f1:6.3f} {prec:10.3f} {rec:8.3f} {support:8d}")


# ─── MAIN TRAINING LOOP ─────────────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping to stop training when validation stops improving.
    
    Saves the best model and stops training when F1 hasn't improved
    for `patience` consecutive epochs.
    """
    
    def __init__(self, patience=7, min_delta=0.001):
        """
        Args:
            patience: Epochs to wait for improvement before stopping
            min_delta: Minimum improvement required to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_f1):
        """
        Check if training should stop.
        
        Args:
            val_f1: Current validation F1 score
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0
        
        return self.early_stop


def train(args):
    """Main training function."""
    backbone = args.backbone
    print("=" * 60)
    print(f"DenseNet121 Chest X-Ray Training  [backbone={backbone}]")
    print("=" * 60)

    # Checkpoint paths (use save_prefix to avoid overwriting)
    prefix = args.save_prefix
    best_path = CHECKPOINT_DIR / f"{prefix}best_model.pth"
    last_path = CHECKPOINT_DIR / f"{prefix}last_model.pth"

    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader, label_names = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        backbone=backbone,
        balanced_sampling=args.balanced_sampling,
        pseudo_csv=getattr(args, "pseudo_csv", None),
        pseudo_dir=getattr(args, "pseudo_dir", None),
        pseudo_weight=getattr(args, "pseudo_weight", 0.5),
    )

    # Create model
    print("\n--- Creating Model ---")
    model = create_model(
        num_labels=len(label_names),
        backbone=backbone,
        xrv_checkpoint=args.xrv_checkpoint,
    )
    
    # Create two-head loss with class weights
    from dataset import load_and_merge_data, build_label_encoder
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = load_and_merge_data()
    label_names_full, label_vectors = build_label_encoder(df)
    
    # Get training split labels for weight calculation
    unique_uids = df["uid"].unique()
    trainval_uids, _ = train_test_split(unique_uids, test_size=0.3, random_state=42)
    train_uids, _ = train_test_split(trainval_uids, test_size=0.214, random_state=42)
    
    train_mask = df["uid"].isin(train_uids)
    train_labels = label_vectors[train_mask]
    
    loss_fn, all_weights = create_two_head_loss(
        train_labels,
        max_pos_weight=args.max_pos_weight,
        anatomy_loss_weight=args.anatomy_loss_weight,
        use_focal=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    
    # Differential learning rates: backbone moves slowly, heads get higher LR.
    # For XRV (CXR-pretrained), use higher backbone LR since features are already useful.
    backbone_lr = args.backbone_lr
    freeze_epochs = args.freeze_epochs
    if backbone == "xrv":
        if args.backbone_lr == DEFAULT_BACKBONE_LR:
            backbone_lr = XRV_BACKBONE_LR
        if args.freeze_epochs == DEFAULT_FREEZE_EPOCHS:
            freeze_epochs = XRV_FREEZE_EPOCHS
        print(f"  XRV backbone: backbone_lr={backbone_lr}, freeze_epochs={freeze_epochs}")

    backbone_params = [p for n, p in model.named_parameters() if n.startswith("features")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("features")]
    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay
    )

    # Optionally keep the backbone fully frozen for the first few epochs
    backbone_frozen = freeze_epochs > 0
    if backbone_frozen:
        set_backbone_trainable(model, False)
        print(f"\nBackbone frozen for the first {freeze_epochs} epoch(s).")
    
    # Warmup scheduler: ramp LR linearly for the first few epochs
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_iters = min(5, args.epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    # We'll manage warmup manually in the epoch loop since ReduceLROnPlateau
    # doesn't compose with SequentialLR easily.
    
    if HAS_TENSORBOARD:
        writer = SummaryWriter(f"runs/densenet121_{time.strftime('%Y%m%d_%H%M%S')}")
    else:
        writer = None
        print("  (TensorBoard not available, skipping logging)")
    
    early_stopping = EarlyStopping(patience=args.patience)

    # Set up CutMix / MixUp augmentation
    mix_fn = None
    if hasattr(args, 'cutmix') and args.cutmix:
        mix_fn = CutMix(alpha=1.0)
        print("  CutMix augmentation enabled (alpha=1.0)")
    elif (hasattr(args, 'mixup') and args.mixup) and not (hasattr(args, 'no_mixup') and args.no_mixup):
        mix_fn = MixUp(alpha=0.2)
        print("  MixUp augmentation enabled (alpha=0.2)")

    # Training loop — report pathology-only F1 as the headline metric
    print(f"\n--- Training for up to {args.epochs} epochs (early stopping patience={args.patience}) ---")
    print(f"{'Ep':>4} | {'TrLoss':>8} {'TrP':>7} {'TrA':>7} | {'VaLoss':>8} {'VaP':>7} {'VaA':>7} | {'PaF1':>6} {'PaAUC':>6} | {'AnF1':>6} {'AnAUC':>6} | {'LR':>10}")
    print("-" * 115)
    
    best_val_path_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        if backbone_frozen and epoch > freeze_epochs:
            set_backbone_trainable(model, True)
            backbone_frozen = False
            print(f"  >> Backbone unfrozen at epoch {epoch}")
        
        # Warmup: ramp up LR linearly for the first few epochs
        if warmup_scheduler is not None and epoch <= warmup_iters:
            warmup_scheduler.step()

        # Train
        train_loss, train_path_loss, train_anat_loss, train_f1, train_prec, train_rec = train_one_epoch(
            model, train_loader, loss_fn, optimizer, epoch, mix_fn=mix_fn
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn)
        
        # Headline metric: pathology-only micro F1
        val_path_f1 = val_metrics.get("path_f1_micro", val_metrics["f1_micro"])
        val_path_auc = val_metrics.get("path_auc", val_metrics["auc"])
        val_anat_f1 = val_metrics.get("anat_f1_micro", 0.0)
        val_anat_auc = val_metrics.get("anat_auc", 0.0)
        val_path_loss = val_metrics.get("path_loss", 0.0)
        val_anat_loss = val_metrics.get("anat_loss", 0.0)
        
        # Only start reducing LR after warmup is done
        if epoch > warmup_iters:
            reduce_scheduler.step(val_path_f1)
        current_lr = optimizer.param_groups[-1]["lr"]
        
        elapsed = time.time() - start_time
        print(
            f"{epoch:4d} | {train_loss:8.4f} {train_path_loss:7.4f} {train_anat_loss:7.4f} | "
            f"{val_metrics['loss']:8.4f} {val_path_loss:7.4f} {val_anat_loss:7.4f} | "
            f"{val_path_f1:6.4f} {val_path_auc:6.4f} | "
            f"{val_anat_f1:6.4f} {val_anat_auc:6.4f} | "
            f"{current_lr:10.6f} | {elapsed:.1f}s"
        )
        
        if writer is not None:
            writer.add_scalars("Loss", {"total": train_loss, "pathology": train_path_loss, "anatomy": train_anat_loss}, epoch)
            writer.add_scalars("F1/pathology", {"train": train_f1, "val": val_path_f1}, epoch)
            writer.add_scalar("F1/anatomy/val", val_anat_f1, epoch)
            writer.add_scalar("AUC/pathology/val", val_path_auc, epoch)
            writer.add_scalar("LR", current_lr, epoch)
        
        # Save best model — select on pathology F1
        if val_path_f1 > best_val_path_f1:
            best_val_path_f1 = val_path_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_path_f1": val_path_f1,
                "val_path_auc": val_path_auc,
                "val_anat_f1": val_anat_f1,
                "val_anat_auc": val_anat_auc,
                "label_names": label_names,
                "backbone": backbone,
                "xrv_checkpoint": getattr(args, "xrv_checkpoint", None),
                "n_pathology": N_PATHOLOGY,
                "n_anatomy": N_ANATOMY,
            }, best_path)
            print(f"  >> Best model saved (path F1: {val_path_f1:.4f}, path AUC: {val_path_auc:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_path_f1": val_path_f1,
            "label_names": label_names,
            "backbone": backbone,
            "n_pathology": N_PATHOLOGY,
            "n_anatomy": N_ANATOMY,
        }, last_path)
        
        if early_stopping(val_path_f1):
            print(f"\n>> Early stopping triggered at epoch {epoch}")
            print(f">> Best pathology F1: {early_stopping.best_score:.4f}")
            break
    
    if writer is not None:
        writer.close()
    
    # Reload best model, tune thresholds, evaluate
    print("\n--- Loading Best Model For Threshold Tuning & Test Evaluation ---")
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = validate(model, val_loader, loss_fn)
    thresholds = find_optimal_thresholds(val_metrics["probs"], val_metrics["labels"], label_names)

    checkpoint["thresholds"] = thresholds
    torch.save(checkpoint, best_path)

    # ── Phase 4: Separate reporting ──
    print("\n" + "=" * 60)
    print("RESULTS — Pathology Head (used for hallucination detection)")
    print("=" * 60)
    print_per_class_metrics(
        val_metrics["probs"][:, :N_PATHOLOGY],
        val_metrics["labels"][:, :N_PATHOLOGY],
        thresholds[:N_PATHOLOGY],
        PATHOLOGY_LABELS, "Validation (Pathology)"
    )

    if N_ANATOMY > 0:
        print("\n" + "=" * 60)
        print("RESULTS — Anatomy Head (auxiliary, not used downstream)")
        print("=" * 60)
        print_per_class_metrics(
            val_metrics["probs"][:, N_PATHOLOGY:],
            val_metrics["labels"][:, N_PATHOLOGY:],
            thresholds[N_PATHOLOGY:],
            ANATOMY_LABELS, "Validation (Anatomy)"
        )

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    test_metrics = validate(model, test_loader, loss_fn, thresholds=thresholds)

    print(f"\n--- Headline Metrics (Pathology Only — for report) ---")
    print(f"  Pathology F1 (micro): {test_metrics.get('path_f1_micro', test_metrics['f1_micro']):.4f}")
    print(f"  Pathology F1 (macro): {test_metrics.get('path_f1_macro', 0):.4f}")
    print(f"  Pathology AUC:        {test_metrics.get('path_auc', test_metrics['auc']):.4f}")
    print(f"  Total Loss:           {test_metrics['loss']:.4f}")
    print(f"  Pathology Loss:       {test_metrics.get('path_loss', 0):.4f}")
    if N_ANATOMY > 0:
        print(f"  Anatomy Loss:         {test_metrics.get('anat_loss', 0):.4f}")
        print(f"\n--- Auxiliary Metrics (Anatomy — not used downstream) ---")
        print(f"  Anatomy F1 (micro):   {test_metrics.get('anat_f1_micro', 0):.4f}")
        print(f"  Anatomy AUC:          {test_metrics.get('anat_auc', 0):.4f}")

    print_per_class_metrics(
        test_metrics["probs"][:, :N_PATHOLOGY],
        test_metrics["labels"][:, :N_PATHOLOGY],
        thresholds[:N_PATHOLOGY],
        PATHOLOGY_LABELS, "Test (Pathology)"
    )
    if N_ANATOMY > 0:
        print_per_class_metrics(
            test_metrics["probs"][:, N_PATHOLOGY:],
            test_metrics["labels"][:, N_PATHOLOGY:],
            thresholds[N_PATHOLOGY:],
            ANATOMY_LABELS, "Test (Anatomy)"
        )

    print(f"\nBest model saved to: {best_path}")
    print(f"  Includes: tuned thresholds, label names, head metadata")
    print(f"  Backbone: {backbone}")
    print("Training complete!")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet121 Two-Head on Chest X-Ray")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Learning rate for the classifier heads")
    parser.add_argument("--backbone_lr", type=float, default=DEFAULT_BACKBONE_LR,
                        help="Learning rate for the pretrained backbone")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pseudo_csv", type=str, default=None,
                        help="Path to a pseudo-label manifest from "
                             "src/pseudo/generate_pseudo_labels.py (e.g. data/pseudo_labels.csv). "
                             "Pseudo-labeled REAL pool images are added to the TRAIN split only; "
                             "val/test stay real-only.")
    parser.add_argument("--pseudo_dir", type=str, default=None,
                        help="Folder the pseudo manifest's filenames are relative to "
                             "(default: data/pseudo_pool)")
    parser.add_argument("--pseudo_weight", type=float, default=0.5,
                        help="Relative training weight of pseudo-labeled vs real rows "
                             "(real=1.0; default pseudo=0.5 hedges against label noise)")
    parser.add_argument("--max_pos_weight", type=float, default=DEFAULT_MAX_POS_WEIGHT)
    parser.add_argument("--freeze_epochs", type=int, default=DEFAULT_FREEZE_EPOCHS,
                        help="Epochs to keep backbone frozen (0 to disable)")
    parser.add_argument("--backbone", type=str, default="imagenet",
                        choices=["imagenet", "xrv", "resnet50"],
                        help="Backbone architecture: imagenet (DenseNet121), xrv (XRV DenseNet), resnet50")
    parser.add_argument("--xrv_checkpoint", type=str, default="densenet121-res224-chex")
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument("--anatomy_loss_weight", type=float, default=0.3,
                        help="Weight for the anatomy auxiliary loss (0 to disable)")
    parser.add_argument("--balanced_sampling", action="store_true",
                        help="Use class-balanced sampling to oversample rare classes")
    parser.add_argument("--focal_loss", action="store_true",
                        help="Use focal loss instead of standard BCE (helps with class imbalance)")
    parser.add_argument("--focal_alpha", type=float, default=DEFAULT_FOCAL_ALPHA,
                        help="Focal loss alpha (weighting factor)")
    parser.add_argument("--focal_gamma", type=float, default=DEFAULT_FOCAL_GAMMA,
                        help="Focal loss gamma (focusing parameter, higher=more focus on hard examples)")
    parser.add_argument("--label_smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING,
                        help="Label smoothing factor (0.0=disabled, 0.05=recommended)")
    parser.add_argument("--cutmix", action="store_true",
                        help="Enable CutMix augmentation during training")
    parser.add_argument("--mixup", action="store_true", default=True,
                        help="Enable MixUp augmentation during training (default: True)")
    parser.add_argument("--no_mixup", action="store_true",
                        help="Disable MixUp augmentation")

    args = parser.parse_args()
    train(args)