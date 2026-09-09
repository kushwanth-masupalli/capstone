import torch
import numpy as np
import cv2
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, class_idx: int, output_path: str) -> str:
    """Generate a Grad‑CAM heat‑map overlay for ``class_idx``.

    Parameters
    ----------
    model: torch.nn.Module
        The classifier model (already in eval mode).
    image_tensor: torch.Tensor
        Input tensor of shape (1, 1, H, W) as used for the classifier.
    class_idx: int
        Index of the target class (matching ``model.pathologies`` order).
    output_path: str
        File path where the resulting PNG overlay will be saved.
    """
    # Identify a suitable target layer. For TorchXRayVision DenseNet the last
    # dense block is a good choice.
    # We attempt a few common attribute names; fall back to the entire features
    # module if none are found.
    target_layer = getattr(model, "features", None)
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        target_layer = model.features.denseblock4

    # Initialise GradCAM with the chosen layer.
    cam = GradCAM(model, target_layer)
    # GradCAM returns a grayscale cam of shape (1, H, W).
    targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets # type: ignore[arg-type]
    )
    cam_np = grayscale_cam[0]
    # Normalise cam to [0, 255]
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    cam_np = np.uint8(255 * cam_np)

    # Convert the original grayscale image to RGB for overlay.
    img_np = image_tensor.squeeze().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = np.uint8(255 * img_np)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Apply colour map and blend.
    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, 0.4, img_rgb, 0.6, 0)

    # Write out the overlay image.
    cv2.imwrite(output_path, overlay)
    return output_path
