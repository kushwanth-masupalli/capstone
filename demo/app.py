#!/usr/bin/env python3
"""Streamlit demo for the chest‑X‑ray report generator.

Run with:
    streamlit run demo/app.py

The app lets a user upload a frontal X‑ray image, optionally type patient history,
presses **Generate** and shows the generated report, classifier predictions,
Grad‑CAM overlays, and hallucination flags.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make the project ``src`` folder importable.   
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

# Import helper modules that implement the pipeline steps directly
# (no subprocess — avoids interpreter/env mismatches).
from classifier.predict import load_classifier, predict, preprocess_image
from explainability.gradcam import generate_gradcam
from inference.pipeline import run_report
from hallucination.check import extract_findings, compare_hallucination

st.title("Chest X‑Ray Report Generator (Qwen‑2‑VL‑2B LoRA)")

uploaded = st.file_uploader("Upload a frontal X‑ray image (PNG/JPG)", type=["png", "jpg", "jpeg"])
history_text = st.text_area("Optional patient history (free‑form text)", height=150)

if uploaded:
    # Save uploaded file to a temporary location.
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp_file:
        tmp_file.write(uploaded.getvalue())
        img_path = tmp_file.name
    st.image(img_path, caption="Uploaded image", width=400)

    if st.button("Generate report"):
        try:
            # ---------- Classifier ----------
            classifier_probs = predict(img_path)  # dict {label: prob}
            prob_rows = []
            predicted_labels = []
            for label, prob in classifier_probs.items():
                is_pred = prob >= 0.5
                prob_rows.append({"Finding": label, "Probability": f"{prob:.3f}", "Predicted": "✅" if is_pred else ""})
                if is_pred:
                    predicted_labels.append(label)
            st.subheader("Classifier Predictions")
            st.table(prob_rows)

            # ---------- Grad‑CAM ----------
            if predicted_labels:
                st.subheader("Grad‑CAM visualizations (top predictions)")
                classifier_model = load_classifier()
                img_tensor = preprocess_image(Path(img_path))
                label_to_idx = {lbl: idx for idx, lbl in enumerate(classifier_model.pathologies)}
                for label in predicted_labels[:2]:
                    class_idx = label_to_idx.get(label)
                    if class_idx is None:
                        continue
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as cam_file:
                        cam_path = cam_file.name
                    try:
                        generate_gradcam(classifier_model, img_tensor, class_idx, cam_path)
                        st.image(cam_path, caption=f"Grad‑CAM for {label}", width=400)
                    finally:
                        if os.path.exists(cam_path):
                            os.remove(cam_path)
            else:
                st.info("No findings predicted with probability ≥ 0.5 – Grad‑CAM skipped.")

            # ---------- Report Generation ----------
            generated_report = run_report(img_path)
            st.subheader("Generated Report")
            st.write(generated_report)

            # ---------- Hallucination Detection ----------
            vocab = list(classifier_probs.keys())
            mentioned = extract_findings(generated_report, vocab)
            hallucination_messages = compare_hallucination(set(predicted_labels), mentioned)
            if hallucination_messages:
                st.subheader("Hallucination Flags")
                for msg in hallucination_messages:
                    st.warning(msg)
            else:
                st.success("No hallucinations detected.")

        except Exception as e:
            st.error("Model inference failed")
            st.exception(e)

        finally:
            # Clean up temporary uploaded image.
            if os.path.exists(img_path):
                os.remove(img_path)
else:
    st.info("Upload an image to get started.")