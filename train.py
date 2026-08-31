"""
DenseNet121 Chest X-Ray Training Script
=========================================
Trains a DenseNet121 model on Indiana Chest X-Ray dataset for multi-label
classification of chest pathology.

Usage:
    python train.py                    # Default settings
    python train.py --epochs 30        # Custom epochs
    python train.py --batch_size 32    # Custom batch size
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
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4       # LR for the new classifier head
DEFAULT_BACKBONE_LR = 1e-5         # Lower LR for the pretrained backbone
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_PATIENCE = 7  # Early stopping patience
DEFAULT_MAX_POS_WEIGHT = 10.0      # Cap on BCE pos_weight for rare classes
DEFAULT_FREEZE_EPOCHS = 2          # Epochs to keep backbone frozen at the start

# Model
NUM_LABELS = 15  # Top 15 problems
PRETRAINED = True  # Use ImageNet pretrained weights

# Checkpointing
CHECKPOINT_DIR = Path("checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── MODEL ───────────────────────────────────────────────────────────────────

def create_model(num_labels=NUM_LABELS, pretrained=PRETRAINED):
    """
    Create DenseNet121 model with custom output layer for multi-label classification.
    
    Architecture:
        - DenseNet121 backbone (pretrained on ImageNet)
        - Replace final classifier layer with custom layer for 15 labels
        - No sigmoid (handled by BCEWithLogitsLoss)
    
    Args:
        num_labels: Number of output labels
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        model: PyTorch model
    """
    from torchvision import models
    
    # Load pretrained DenseNet121
    if pretrained:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        model = models.densenet121(weights=None)
    
    # Get the number of features in the final layer
    num_features = model.classifier.in_features
    
    # Replace classifier with custom layer
    # No sigmoid — BCEWithLogitsLoss handles it internally
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, num_labels)
    )
    
    print(f"Model: DenseNet121")
    print(f"  Backbone: {'Pretrained ImageNet' if pretrained else 'Random init'}")
    print(f"  Features: {num_features}")
    print(f"  Output: {num_labels} labels (no sigmoid)")
    print(f"  Device: {DEVICE}")
    
    return model.to(DEVICE)


def set_backbone_trainable(model, trainable):
    """
    Freeze or unfreeze every parameter outside the classifier head.

    Used to keep the pretrained DenseNet features stable for the first few
    epochs while the randomly-initialized classifier head catches up, before
    letting gradients flow into the backbone. Reduces early-training
    distortion of the pretrained features on a small dataset.
    """
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = trainable


# ─── LOSS FUNCTION ───────────────────────────────────────────────────────────

def create_loss_function(label_vectors_train, max_pos_weight=DEFAULT_MAX_POS_WEIGHT):
    """
    Create BCEWithLogitsLoss with (capped) class weights for imbalanced
    multi-label data.
    
    The 'normal' class has 1,387 images while smaller classes have ~100.
    Class weights help the model not ignore rare conditions — but uncapped
    inverse-frequency weighting produces weights above 20x for the rarest
    classes, which tends to push the model toward over-predicting positives
    (hurting precision) and can destabilize training. Capping keeps the
    rebalancing effect without letting a handful of classes dominate the
    loss.
    
    Args:
        label_vectors_train: np.ndarray of training labels for weight calculation
        max_pos_weight: upper bound applied to any single class's weight
        
    Returns:
        criterion: Loss function
    """
    # Calculate class weights (inverse frequency)
    pos_counts = label_vectors_train.sum(axis=0)
    total = label_vectors_train.shape[0]
    
    # Weight = total / (2 * positive_count)
    # This gives higher weight to underrepresented classes
    weights = total / (2 * pos_counts + 1e-6)
    weights = np.clip(weights, a_min=None, a_max=max_pos_weight)
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    
    print(f"\nClass weights:")
    from dataset import load_and_merge_data, build_label_encoder
    df = load_and_merge_data()
    label_names, _ = build_label_encoder(df)
    for i, name in enumerate(label_names):
        print(f"  {name:25s} pos={int(pos_counts[i]):5d}  weight={weights[i]:.2f}")
    
    return criterion


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients: pos_weight terms (up to DEFAULT_MAX_POS_WEIGHT) can
        # produce large gradient spikes on batches with rare-class positives.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        
        # Convert logits to binary predictions (threshold=0.0 since no sigmoid)
        preds = (outputs > 0.0).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    # Calculate epoch metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    avg_loss = total_loss / len(train_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)
    
    return avg_loss, f1, precision, recall


def validate(model, val_loader, criterion, thresholds=None):
    """
    Validate the model.

    Args:
        thresholds: optional np.ndarray of shape (n_labels,) with a
            per-class decision threshold. Defaults to 0.5 for every class
            when not given (matches original behavior).

    Returns a dict of metrics plus the raw probs/labels, so callers can
    reuse them for threshold tuning or per-class reporting without a second
    forward pass over the data.
    """
    model.eval()
    
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    if thresholds is None:
        thresholds = np.full(all_probs.shape[1], 0.5, dtype=np.float32)
    all_preds = (all_probs > thresholds[np.newaxis, :]).astype(np.float32)
    
    avg_loss = total_loss / len(val_loader.dataset)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)
    
    # Calculate AUC (with safety check for single-class)
    try:
        auc = roc_auc_score(all_labels, all_probs, average="micro")
    except ValueError:
        auc = 0.0
    
    return {
        "loss": avg_loss,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "probs": all_probs,
        "labels": all_labels,
    }


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
    print("=" * 60)
    print("DenseNet121 Chest X-Ray Training")
    print("=" * 60)
    
    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader, label_names = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\n--- Creating Model ---")
    model = create_model(num_labels=len(label_names))
    
    # Create loss function with class weights
    # We need to get the training label vectors for weight calculation
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
    
    criterion = create_loss_function(train_labels, max_pos_weight=args.max_pos_weight)
    
    # Differential learning rates: the pretrained backbone moves slowly so we
    # don't distort its ImageNet features before the new classifier head has
    # learned anything; the head gets the higher LR since it starts random.
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay
    )

    # Optionally keep the backbone fully frozen for the first few epochs so
    # only the classifier head trains initially.
    backbone_frozen = args.freeze_epochs > 0
    if backbone_frozen:
        set_backbone_trainable(model, False)
        print(f"\nBackbone frozen for the first {args.freeze_epochs} epoch(s).")
    
    # Learning rate scheduler — watches val F1 (what we actually select on),
    # not val loss, since the two can diverge under heavy pos_weight.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    
    # Tensorboard writer (optional)
    if HAS_TENSORBOARD:
        writer = SummaryWriter(f"runs/densenet121_{time.strftime('%Y%m%d_%H%M%S')}")
    else:
        writer = None
        print("  (TensorBoard not available, skipping logging)")
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    print(f"\n--- Training for up to {args.epochs} epochs (early stopping patience={args.patience}) ---")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val F1(mi)':>10} | {'Val F1(ma)':>10} | {'Val AUC':>8} | {'LR':>10}")
    print("-" * 95)
    
    best_val_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Unfreeze the backbone once the warmup window has passed
        if backbone_frozen and epoch > args.freeze_epochs:
            set_backbone_trainable(model, True)
            backbone_frozen = False
            print(f"  >> Backbone unfrozen at epoch {epoch}")
        
        # Train
        train_loss, train_f1, train_precision, train_recall = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion)
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1_micro"]
        val_f1_macro = val_metrics["f1_macro"]
        val_auc = val_metrics["auc"]
        
        # Update learning rate based on the metric we actually select on
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[-1]["lr"]  # head LR
        
        # Print epoch results
        elapsed = time.time() - start_time
        print(
            f"{epoch:6d} | {train_loss:10.4f} | {val_loss:10.4f} | "
            f"{val_f1:10.4f} | {val_f1_macro:10.4f} | {val_auc:8.4f} | {current_lr:10.6f} | {elapsed:.1f}s"
        )
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("F1", {"train": train_f1, "val_micro": val_f1, "val_macro": val_f1_macro}, epoch)
            writer.add_scalar("AUC/val", val_auc, epoch)
            writer.add_scalar("LR", current_lr, epoch)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_f1_macro": val_f1_macro,
                "val_auc": val_auc,
                "label_names": label_names,
            }, BEST_MODEL_PATH)
            print(f"  >> Best model saved (F1: {val_f1:.4f}, macro F1: {val_f1_macro:.4f})")
        
        # Save last model
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "label_names": label_names,
        }, LAST_MODEL_PATH)
        
        # Early stopping check
        if early_stopping(val_f1):
            print(f"\n>> Early stopping triggered at epoch {epoch}")
            print(f">> Best validation F1: {early_stopping.best_score:.4f}")
            break
    
    if writer is not None:
        writer.close()
    
    # Reload best model, then tune per-class thresholds on the validation set
    print("\n--- Loading Best Model For Threshold Tuning & Test Evaluation ---")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = validate(model, val_loader, criterion)
    thresholds = find_optimal_thresholds(val_metrics["probs"], val_metrics["labels"], label_names)

    # Persist the tuned thresholds alongside the checkpoint so inference code
    # doesn't have to guess at 0.5 later.
    checkpoint["thresholds"] = thresholds
    torch.save(checkpoint, BEST_MODEL_PATH)

    print_per_class_metrics(val_metrics["probs"], val_metrics["labels"], thresholds, label_names, "Validation")

    # Final evaluation on test set, using the tuned thresholds
    print("\n--- Final Test Evaluation ---")
    test_metrics = validate(model, test_loader, criterion, thresholds=thresholds)

    print(f"Test Loss:      {test_metrics['loss']:.4f}")
    print(f"Test F1 (micro): {test_metrics['f1_micro']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test AUC:       {test_metrics['auc']:.4f}")

    print_per_class_metrics(test_metrics["probs"], test_metrics["labels"], thresholds, label_names, "Test")
    
    print(f"\nBest model saved to: {BEST_MODEL_PATH} (includes tuned per-class thresholds)")
    print("Training complete!")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet121 on Chest X-Ray")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate for the classifier head")
    parser.add_argument("--backbone_lr", type=float, default=DEFAULT_BACKBONE_LR, help="Learning rate for the pretrained backbone")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max_pos_weight", type=float, default=DEFAULT_MAX_POS_WEIGHT, help="Cap on BCE pos_weight for rare classes")
    parser.add_argument("--freeze_epochs", type=int, default=DEFAULT_FREEZE_EPOCHS, help="Epochs to keep backbone frozen at the start (0 to disable)")
    
    args = parser.parse_args()
    train(args)