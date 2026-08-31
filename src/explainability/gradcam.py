"""
Grad-CAM Explainability for Chest X-Ray Classifier
====================================================
Generates class-discriminative heatmaps showing which image regions
drove a specific predicted finding.

Usage:
    from src.explainability.gradcam import load_model_for_gradcam, generate_gradcam_overlay

    model, label_names, thresholds = load_model_for_gradcam("checkpoints/best_model.pth")
    overlay = generate_gradcam_overlay(model, image_tensor, target_class_idx=0)
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ImageNet normalization (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_gradcam(checkpoint_path="checkpoints/best_model.pth"):
    """
    Load trained DenseNet121 and return model + metadata.

    Returns:
        model: loaded model in eval mode on DEVICE
        label_names: list of class names
        thresholds: per-class decision thresholds
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    label_names = checkpoint["label_names"]
    thresholds = checkpoint.get("thresholds", np.full(len(label_names), 0.5))

    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(num_features, len(label_names)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, label_names, thresholds


def get_target_layer(model):
    """
    Return the last convolutional layer of DenseNet121 for Grad-CAM.

    For DenseNet121, the features flow through:
        features.denseblock4 -> features.norm5 -> classifier

    We use norm5 (the batch norm after the last dense block) because it
    captures the most semantic information before pooling flattens it.
    """
    return model.features.norm5


def preprocess_image(image_path_or_array, size=224):
    """
    Load and preprocess an image for model input.

    Args:
        image_path_or_array: file path (str/Path) or numpy array (H, W, C) or (H, W)
        size: target resize dimension

    Returns:
        rgb_np: numpy array (H, W, 3) in [0, 1] for CAM overlay
        tensor: preprocessed tensor (1, 3, H, W) ready for model
    """
    if isinstance(image_path_or_array, (str, Path)):
        img = Image.open(image_path_or_array).convert("RGB")
        rgb_np = np.array(img).astype(np.float32) / 255.0
    else:
        rgb_np = image_path_or_array.astype(np.float32)
        if rgb_np.max() > 1.0:
            rgb_np = rgb_np / 255.0
        if rgb_np.ndim == 2:
            rgb_np = np.stack([rgb_np] * 3, axis=-1)
        img = Image.fromarray((rgb_np * 255).astype(np.uint8))

    img_resized = img.resize((size, size), Image.BILINEAR)
    rgb_np = np.array(img_resized).astype(np.float32) / 255.0

    # Normalize for model input
    tensor = (rgb_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    return rgb_np, tensor


def generate_gradcam_overlay(model, image_tensor, rgb_np, target_class_idx,
                              target_layer=None):
    """
    Generate a Grad-CAM heatmap overlay for a specific class.

    Args:
        model: trained model in eval mode
        image_tensor: preprocessed input tensor (1, 3, H, W)
        rgb_np: original RGB image in [0, 1] for overlay background
        target_class_idx: index of the class to explain
        target_layer: override layer (default: last conv layer)

    Returns:
        overlay: numpy array (H, W, 3) with heatmap overlaid on image
        cam: raw CAM values (H, W) in [0, 1]
    """
    if target_layer is None:
        target_layer = get_target_layer(model)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor,
                            targets=None,
                            aug_smooth=None,
                            eigen_smooth=False)

        # grayscale_cam is (1, H, W) — take the single sample
        cam_result = grayscale_cam[0]

        # Overlay on the original image
        overlay = show_cam_on_image(rgb_np, cam_result, use_rgb=True)

    return overlay, cam_result


def predict_with_confidence(model, tensor, label_names, thresholds):
    """
    Run inference and return predictions with confidence.

    Returns:
        list of dicts sorted by probability descending:
        [{"name": "...", "prob": 0.95, "threshold": 0.5, "flagged": True}, ...]
    """
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = []
    for i, name in enumerate(label_names):
        results.append({
            "name": name,
            "prob": float(probs[i]),
            "threshold": float(thresholds[i]),
            "flagged": bool(probs[i] > thresholds[i]),
        })

    results.sort(key=lambda x: x["prob"], reverse=True)
    return results


def generate_gradcam_for_all_classes(model, image_path_or_array, label_names,
                                      thresholds, min_prob=0.1):
    """
    Generate Grad-CAM overlays for all classes above a probability threshold.

    Args:
        model: trained model
        image_path_or_array: image source
        label_names: list of class names
        thresholds: per-class thresholds
        min_prob: minimum probability to generate a heatmap for

    Returns:
        list of dicts: [{"name": ..., "prob": ..., "overlay": np.array, "cam": np.array}, ...]
    """
    rgb_np, tensor = preprocess_image(image_path_or_array)
    predictions = predict_with_confidence(model, tensor, label_names, thresholds)

    results = []
    for pred in predictions:
        if pred["prob"] < min_prob:
            continue
        class_idx = label_names.index(pred["name"])
        overlay, cam = generate_gradcam_overlay(model, tensor, rgb_np, class_idx)
        results.append({
            "name": pred["name"],
            "prob": pred["prob"],
            "threshold": pred["threshold"],
            "flagged": pred["flagged"],
            "overlay": overlay,
            "cam": cam,
        })

    return results, rgb_np
