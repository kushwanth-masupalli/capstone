#!/usr/bin/env python3
"""
Generate training CSV for LoRA finetuning (Plan B).

- Joins ``indiana_reports.csv`` and ``indiana_projections.csv`` on ``uid``.
- Keeps only frontal views (``projection == "Frontal"``).
- Produces three columns required by ``finetune.py``:
    * ``uid`` – optional identifier (kept for bookkeeping).
    * ``image_path`` – absolute or relative path to the **pre‑processed** PNG (``data/images/preprocessed/<filename>``).
    * ``report_text`` – the ground‑truth ``findings`` text that the model should learn to generate.

The resulting CSV is saved as ``data/train_finetune.csv`` and can be fed directly to the finetuning script.
"""

import os
import pandas as pd

def main() -> None:
    reports_path = "data/indiana_reports.csv"
    projections_path = "data/indiana_projections.csv"
    out_path = "data/train_finetune.csv"
    preproc_dir = "data/images/preprocessed"

    # Load CSVs
    reports = pd.read_csv(reports_path)
    projections = pd.read_csv(projections_path)

    # Keep only frontal images
    frontal = projections[projections["projection"] == "Frontal"].copy()

    # Merge to get the findings column (ground‑truth report text)
    merged = frontal.merge(reports[["uid", "findings"]], on="uid", how="left")

    # Build full image path to the pre‑processed image
    merged["image_path"] = merged["filename"].apply(lambda fn: os.path.join(preproc_dir, fn))

    # Rename column to match finetune.py expectations
    merged = merged.rename(columns={"findings": "report_text"})

    # Keep only the columns we need
    finetune_csv = merged[["uid", "image_path", "report_text"]].copy()

    # Drop any rows with missing report text (should be rare)
    finetune_csv = finetune_csv.dropna(subset=["report_text"]).reset_index(drop=True)

    finetune_csv.to_csv(out_path, index=False)
    print(f"✅ Generated {len(finetune_csv)} training rows → {out_path}")

if __name__ == "__main__":
    main()
