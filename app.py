"""Quick test interface with Grad-CAM. Run with: python app.py"""

import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from flask import Flask, request, render_template_string
from PIL import Image
from torchvision import models

try:
    import torchxrayvision as xrv
    HAS_XRV = True
except ImportError:
    HAS_XRV = False

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# ── Two-head classifier (must match train.py) ────────────────────────────────

class TwoHeadClassifier(nn.Module):
    def __init__(self, features_module, n_pathology=10, n_anatomy=5):
        super().__init__()
        self.features = features_module
        self.n_pathology = n_pathology
        self.n_anatomy = n_anatomy
        self.pathology_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1024, n_pathology),
        )
        self.anatomy_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1024, n_anatomy),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = F.relu(feat)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)
        path_out = self.pathology_head(feat)
        anat_out = self.anatomy_head(feat)
        return torch.cat([path_out, anat_out], dim=1)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    label_names = checkpoint["label_names"]
    thresholds = checkpoint.get("thresholds", np.full(len(label_names), 0.5))
    backbone = checkpoint.get("backbone", "imagenet")
    n_pathology = checkpoint.get("n_pathology", 10)
    n_anatomy = checkpoint.get("n_anatomy", 5)

    print(f"Loading model: backbone={backbone}, labels={len(label_names)}")

    if backbone == "xrv":
        assert HAS_XRV, "torchxrayvision is required for this checkpoint"
        xrv_checkpoint = checkpoint.get("xrv_checkpoint", "densenet121-res224-chex")
        xrv_model = xrv.models.DenseNet(weights=xrv_checkpoint)
        model = TwoHeadClassifier(xrv_model.features, n_pathology, n_anatomy)
    else:
        densenet = models.densenet121(weights=None)
        model = TwoHeadClassifier(densenet.features, n_pathology, n_anatomy)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Pick the last conv layer in denseblock4 for Grad-CAM
    target_layer = model.features.denseblock4.denselayer16.conv2

    print(f"  Label names: {label_names}")
    print(f"  Thresholds:  {thresholds}")
    print(f"  Device:      {DEVICE}")
    return model, label_names, thresholds, target_layer, backbone


model, label_names, thresholds, target_layer, backbone = load_model()

# ── Preprocessing ─────────────────────────────────────────────────────────────

if backbone == "xrv" and HAS_XRV:
    _xrv_resizer = xrv.datasets.XRayResizer(224)
    _xrv_cropper = xrv.datasets.XRayCenterCrop()

    def preprocess(img: Image.Image):
        """XRV pipeline: grayscale → normalize → resize → crop → tensor (1,1,224,224)."""
        gray = img.convert("L")
        arr = np.array(gray).astype(np.float32)[np.newaxis, ...]  # (1,H,W)
        arr = xrv.datasets.normalize(arr, maxval=255.0)
        arr = _xrv_resizer(arr)
        arr = _xrv_cropper(arr)
        return torch.from_numpy(arr).float().unsqueeze(0).to(DEVICE)  # (1,1,224,224)

    def to_rgb_np_for_cam(img: Image.Image):
        """Return 224×224 float32 RGB array [0,1] for Grad-CAM overlay."""
        return np.array(img.resize((224, 224))).astype(np.float32) / 255.0
else:
    from torchvision import transforms
    _eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def preprocess(img: Image.Image):
        return _eval_transform(img).unsqueeze(0).to(DEVICE)

    def to_rgb_np_for_cam(img: Image.Image):
        return np.array(img.resize((224, 224))).astype(np.float32) / 255.0


def img_to_b64(arr):
    """Convert numpy RGB array [0,1] to base64 PNG string."""
    im = (arr * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    cam_images = {}
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            img_tensor = preprocess(img)
            rgb_np = to_rgb_np_for_cam(img)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            results = []
            for i, name in enumerate(label_names):
                results.append((name, float(probs[i]), float(thresholds[i])))
            results.sort(key=lambda x: x[1], reverse=True)

            # Grad-CAM for top 4 predictions
            with GradCAM(model=model, target_layers=[target_layer]) as cam:
                for name, prob, thresh in results[:4]:
                    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
                    overlay = show_cam_on_image(rgb_np, grayscale_cam[0], use_rgb=True)
                    cam_images[name] = img_to_b64(overlay)

    return render_template_string(HTML, results=results, cam_images=cam_images)


HTML = r"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Chest X-Ray Test</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f5f5f5; color: #222; padding: 20px; }
  .wrap { max-width: 620px; margin: 0 auto; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  p.sub { color: #666; font-size: 13px; margin-bottom: 16px; }
  form { margin-bottom: 24px; }
  input[type=file] { font-size: 14px; }
  input[type=submit] { margin-left: 8px; padding: 4px 14px; font-size: 14px;
                       border: 1px solid #aaa; background: #fff; border-radius: 4px;
                       cursor: pointer; }
  input[type=submit]:hover { background: #eee; }
  .results { list-style: none; margin-bottom: 20px; }
  .row { display: flex; align-items: center; margin-bottom: 5px; font-size: 13px; }
  .name { width: 160px; font-weight: 500; }
  .bar-wrap { flex: 1; background: #e0e0e0; height: 16px; border-radius: 3px; overflow: hidden; }
  .bar { height: 100%; background: #4a90d9; border-radius: 3px; }
  .pct { width: 46px; text-align: right; margin-left: 6px; font-size: 12px; color: #444; }
  .flag { color: #c0392b; font-weight: 600; font-size: 11px; margin-left: 3px; }
  hr { border: none; border-top: 1px solid #ddd; margin: 14px 0; }
  .cam-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
  .cam-card { text-align: center; }
  .cam-card img { width: 100%; border-radius: 4px; border: 1px solid #ddd; }
  .cam-card .cam-label { font-size: 12px; color: #555; margin-top: 3px; }
  .note { font-size: 11px; color: #888; margin-top: 10px; }
</style></head>
<body>
<div class="wrap">
  <h1>Chest X-Ray Classifier</h1>
  <p class="sub">Upload a chest X-ray. Grad-CAM shows which regions drove each prediction.</p>

  <form method=post enctype=multipart/form-data>
    <input type=file name=image accept="image/*" required>
    <input type=submit value="Predict">
  </form>

  {% if results %}
  <hr>
  <ul class="results">
  {% for name, prob, thresh in results %}
    <li class="row">
      <span class="name">{{ name }}</span>
      <div class="bar-wrap"><div class="bar" style="width:{{ (prob*100)|int }}%"></div></div>
      <span class="pct">{{ "%.1f"|format(prob*100) }}%</span>
      {% if prob > thresh %}<span class="flag">+</span>{% endif %}
    </li>
  {% endfor %}
  </ul>

  <p class="note">Grad-CAM overlays (top 4 predictions). Red = high importance.</p>
  <div class="cam-grid">
  {% for name, prob, thresh in results[:4] %}
    {% if name in cam_images %}
    <div class="cam-card">
      <img src="data:image/png;base64,{{ cam_images[name] }}" alt="{{ name }}">
      <div class="cam-label">{{ name }} ({{ "%.0f"|format(prob*100) }}%)</div>
    </div>
    {% endif %}
  {% endfor %}
  </div>
  {% endif %}
</div>
</body>
</html>"""

if __name__ == "__main__":
    print(f"Starting server on http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
