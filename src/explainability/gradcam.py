"""Grad-CAM generation for the TorchXRayVision DenseNet classifier.

Given a loaded XRV DenseNet model, a preprocessed input tensor (as produced
by ``classifier.predict.preprocess_image``), and the class index of the
finding to explain, this module computes a Grad-CAM heatmap and writes an
overlay PNG (heatmap blended over the input X-ray) to ``output_path``.
"""

import numpy as np
import torch
import cv2
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def _get_target_layer(model: torch.nn.Module):
    """Return the last convolutional feature layer of the XRV DenseNet.

    TorchXRayVision's DenseNet exposes its feature extractor as
    ``model.features`` (a torchvision-style DenseNet feature block ending in
    ``norm5``). We target ``norm5`` — the final batch-norm applied to the
    last dense block's output, immediately before global pooling — since
    this is the standard Grad-CAM target for DenseNet architectures. Picking
    an earlier or unrelated layer is what causes meaningless heatmaps
    (e.g. activations concentrated at image borders/artifacts instead of
    anatomy).
    """
    features = getattr(model, "features", None)
    if features is None:
        raise AttributeError(
            "Model has no `.features` attribute — cannot locate a conv layer for Grad-CAM."
        )
    if hasattr(features, "norm5"):
        return features.norm5
    # Fallback: last child module of the feature extractor.
    children = list(features.children())
    if not children:
        raise AttributeError("`.features` has no child layers to target for Grad-CAM.")
    return children[-1]


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_idx: int,
    output_path: str,
) -> str:
    """Compute a Grad-CAM overlay for ``class_idx`` and save it to ``output_path``.

    Parameters
    ----------
    model:
        A loaded ``torchxrayvision`` DenseNet classifier (see
        ``classifier.predict.load_classifier``), in eval mode.
    image_tensor:
        The preprocessed input tensor, shape ``(1, 1, 224, 224)``, as
        produced by ``classifier.predict.preprocess_image``. Values are the
        XRV-normalized range (roughly [-1024, 1024]), NOT [0, 1].
    class_idx:
        Index into ``model.pathologies`` for the finding to explain.
    output_path:
        File path (PNG) to write the heatmap overlay to.

    Returns
    -------
    str
        The ``output_path`` that was written, for convenience.
    """
    model.eval()
    target_layer = _get_target_layer(model)

    # Grad-CAM needs gradients w.r.t. the input to backprop through.
    input_tensor = image_tensor.clone().detach().requires_grad_(True)

    # Build a normalized [0, 1] RGB version of the *input* image for the
    # overlay — using the raw XRV-normalized tensor would produce a blank
    # or garbage-looking background since its value range isn't [0, 1].
    with torch.no_grad():
        img_np = input_tensor[0, 0].cpu().numpy()  # (H, W), XRV-normalized range
    img_min, img_max = img_np.min(), img_np.max()
    if img_max - img_min < 1e-6:
        img_vis = np.zeros_like(img_np, dtype=np.float32)
    else:
        img_vis = (img_np - img_min) / (img_max - img_min)
    img_vis_rgb = np.stack([img_vis] * 3, axis=-1).astype(np.float32)  # (H, W, 3) in [0,1]

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # (H, W), values in [0, 1]

    # If the CAM's spatial size doesn't match the display image (can happen
    # depending on target layer stride), resize it to match before overlay.
    if grayscale_cam.shape != img_vis.shape:
        grayscale_cam = cv2.resize(
            grayscale_cam, (img_vis.shape[1], img_vis.shape[0])
        )

    overlay = show_cam_on_image(img_vis_rgb, grayscale_cam, use_rgb=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # show_cam_on_image returns RGB uint8; cv2.imwrite expects BGR.
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return str(out_path)