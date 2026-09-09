import torch
import torchxrayvision as xrv
from pathlib import Path
import numpy as np
from PIL import Image

def load_classifier(weights: str = "densenet121-res224-chex"):
    """Load a TorchXRayVision DenseNet classifier.

    Parameters
    ----------
    weights: str
        Identifier of the pretrained checkpoint to load. The default is the
        CheXpert‑trained checkpoint used throughout the project.
    """
    model = xrv.models.DenseNet(weights=weights)
    model.eval()
    return model

def preprocess_image(image_path: Path):
    """Read an image and apply the TorchXRayVision preprocessing pipeline.

    The XRV models expect a single‑channel (grayscale) image that is normalized,
    resized to 224×224 and centre‑cropped.
    """
    # Load image as grayscale (XRV works with 1‑channel images).
    pil_img = Image.open(image_path).convert("L")
    img_np = np.array(pil_img).astype(np.float32)
    # Add channel dimension
    img_np = img_np[np.newaxis, ...]  # shape (1, H, W)

    # Normalise to [0, 1] using XRV utilities
    img_normalized = xrv.datasets.normalize(img_np, maxval=255.0)
    # Resize to 224×224
    resizer = xrv.datasets.XRayResizer(224)
    img_resized = resizer(img_normalized)
    # Center‑crop (XRV uses a 224×224 crop after resize, this is a no‑op but kept for clarity)
    cropper = xrv.datasets.XRayCenterCrop()
    img_cropped = cropper(img_resized)

    # Convert to torch tensor (B, C, H, W) – XRV models expect a single channel.
    img_tensor = torch.from_numpy(img_cropped).unsqueeze(0).float()
    return img_tensor

def predict(image_path: str):
    """Run the classifier on an image and return a dictionary of label → probability.

    The function loads the model on each call (lightweight for the demo) and
    applies a sigmoid to obtain probabilities for each pathology.
    """
    model = load_classifier()
    img_tensor = preprocess_image(Path(image_path))
    with torch.no_grad():
        logits = model(img_tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    # model.pathologies provides the ordered list of label strings.
    return {label: float(p) for label, p in zip(model.pathologies, probs)}
