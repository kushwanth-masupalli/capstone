import torch
import torch.nn as nn
import torchxrayvision as xrv
from pathlib import Path
import numpy as np
from PIL import Image

# Path to the trained classifier checkpoint (produced by your training run,
# NOT the generic public XRV checkpoint). Change this if you decide a
# different .pth file is your "real" final model.
CHECKPOINT_PATH = Path("checkpoints") / "base_best_model.pth"

# The XRV backbone this checkpoint was fine-tuned from. Only used to build
# the matching feature-extractor architecture before loading your weights —
# the pretrained weights themselves get fully overwritten by your checkpoint.
BASE_XRV_WEIGHTS = "densenet121-res224-chex"


def _build_model_for_checkpoint(num_classes: int) -> nn.Module:
    """Build an XRV DenseNet with a custom classifier head matching the
    trained checkpoint's structure: classifier = Sequential(Dropout, Linear).
    """
    model = xrv.models.DenseNet(weights=BASE_XRV_WEIGHTS)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    # XRV's forward() auto-applies an output normalization step keyed to
    # `op_threshs`, a buffer sized for the ORIGINAL 18-pathology head this
    # backbone shipped with. Our checkpoint replaced the head with a custom
    # 15-class one and never saved its own op_threshs, so this stale buffer
    # is both the wrong size and semantically meaningless here. Disabling it
    # makes forward() skip that step and return raw logits, which is what
    # predict() expects (it applies its own sigmoid).
    model.op_threshs = None
    return model


def load_classifier(checkpoint_path: Path = CHECKPOINT_PATH):
    """Load OUR trained classifier checkpoint (custom 15-class head),
    instead of the generic public XRV checkpoint.

    Returns a model with a `.pathologies` attribute set to the checkpoint's
    own `label_names`, so downstream code (Grad-CAM label lookup, the
    Streamlit table, hallucination checking) automatically uses the correct
    label vocabulary for THIS model rather than XRV's default pathologies.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    label_names = ckpt["label_names"]

    model = _build_model_for_checkpoint(num_classes=len(label_names))
    # strict=False: XRV's DenseNet registers an internal `op_threshs` buffer
    # (per-class decision thresholds) that our training script did not save,
    # since it's unrelated to the actual learned weights. Any OTHER missing
    # or unexpected key would indicate a real architecture mismatch, so we
    # check for that explicitly below instead of silently ignoring everything.
    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    unexpected_missing = [k for k in load_result.missing_keys if k != "op_threshs"]
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected mismatch loading classifier checkpoint.\n"
            f"  Missing (excluding op_threshs): {unexpected_missing}\n"
            f"  Unexpected: {load_result.unexpected_keys}"
        )
    model.eval()

    # Downstream code (app.py, gradcam label-index lookup) expects
    # `model.pathologies` to exist — populate it with our own label set.
    model.pathologies = label_names

    return model


def preprocess_image(image_path: Path):
    """Read an image and apply the TorchXRayVision preprocessing pipeline.

    The XRV backbone expects a single-channel (grayscale) image that is
    normalized, resized to 224x224 and centre-cropped. This preprocessing is
    unchanged by fine-tuning the classifier head, so it still applies as-is.
    """
    pil_img = Image.open(image_path).convert("L")
    img_np = np.array(pil_img).astype(np.float32)
    img_np = img_np[np.newaxis, ...]  # shape (1, H, W)

    img_normalized = xrv.datasets.normalize(img_np, maxval=255.0)
    resizer = xrv.datasets.XRayResizer(224)
    img_resized = resizer(img_normalized)
    cropper = xrv.datasets.XRayCenterCrop()
    img_cropped = cropper(img_resized)

    img_tensor = torch.from_numpy(img_cropped).unsqueeze(0).float()
    return img_tensor


def predict(image_path: str, checkpoint_path: Path = CHECKPOINT_PATH):
    """Run OUR trained classifier on an image and return a dict of
    label -> probability, using the checkpoint's own label vocabulary.

    NOTE: this reloads the model on every call (matches the original
    lightweight-for-demo behavior). If you need this to be fast for
    batch evaluation, load the model once with load_classifier() and
    write a separate batch-prediction loop instead.
    """
    model = load_classifier(checkpoint_path)
    img_tensor = preprocess_image(Path(image_path))
    with torch.no_grad():
        logits = model(img_tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return {label: float(p) for label, p in zip(model.pathologies, probs)}