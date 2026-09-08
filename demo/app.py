#!/usr/bin/env python3
"""Streamlit demo for the chest‑X‑ray report generator.

Run with:
    streamlit run demo/app.py

The app lets a user upload a frontal X‑ray image, optionally type patient history,
presses **Generate** and shows the generated report (and placeholder fields for
findings, Grad‑CAM, hallucination flags).
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

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
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        cmd = ["python", "-m", "src.inference.pipeline", "--image", img_path, "--output", out_path]
        # Run the inference wrapper.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error("Model inference failed")
            st.code(result.stderr)
        else:
            with open(out_path) as f:
                data = json.load(f)
            st.subheader("Generated Report")
            st.write(data.get("generated_report", ""))
            if data.get("predicted_findings"):
                st.subheader("Predicted Findings")
                st.write(", ".join(data["predicted_findings"]))
            if data.get("gradcam_path"):
                st.subheader("Grad‑CAM Overlay")
                st.image(data["gradcam_path"], caption="Grad‑CAM")
            if data.get("hallucination_flags"):
                st.subheader("Hallucination Flags")
                for flag in data["hallucination_flags"]:
                    st.warning(flag)
        # Clean up temp files.
        os.remove(img_path)
        os.remove(out_path)
else:
    st.info("Upload an image to get started.")
