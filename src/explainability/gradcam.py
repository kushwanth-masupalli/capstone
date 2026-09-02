"""
Grad-CAM Explainability for Chest X-Ray Classifier
====================================================

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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_gradcam(checkpoint_path="checkpoints/best_model.pth"):
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
    return model.features.norm5


def preprocess_image(image_path_or_array, size=224):
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

    tensor = (rgb_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    return rgb_np, tensor


def generate_gradcam_overlay(model, image_tensor, rgb_np, target_class_idx, target_layer=None):
    if target_layer is None:
        target_layer = get_target_layer(model)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=None, aug_smooth=None, eigen_smooth=False)
        cam_result = grayscale_cam[0]
        overlay = show_cam_on_image(rgb_np, cam_result, use_rgb=True)
    return overlay, cam_result


def predict_with_confidence(model, tensor, label_names, thresholds):
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


def generate_gradcam_for_all_classes(model, image_path_or_array, label_names, thresholds, min_prob=0.1):
    rgb_np, tensor = preprocess_image(image_path_or_array)
    predictions = predict_with_confidence(model, tensor, label_names, thresholds)

    results = []
    for pred in predictions:
        if pred["prob"] < min_prob:
            continue
        class_idx = label_names.index(pred["name"])
        overlay, cam = generate_gradcam_overlay(model, tensor, rgb_np, class_idx)
        results.append({
            "name": pred["name"], "prob": pred["prob"],
            "threshold": pred["threshold"], "flagged": pred["flagged"],
            "overlay": overlay, "cam": cam,
        })
    return results, rgb_np
