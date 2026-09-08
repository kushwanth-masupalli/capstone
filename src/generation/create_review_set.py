#!/usr/bin/env python3
"""Create a blinded review CSV from the test split and generated reports.

The script samples a fixed number of entries (default 25) from the test CSV,
pairs each image with its reference report and the model‑generated report, and
writes a CSV suitable for a manual reviewer.

Usage:
    python src/generation/create_review_set.py \
        --generated results/finetuned_patient_split/pipeline_results.json \
        --split data/report_splits/test.csv \
        --output results/finetuned_patient_split/review_set.csv \
        [--sample 25]
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Create a blinded review set.")
    parser.add_argument("--generated", required=True, help="JSON file from generate_reports.py")
    parser.add_argument("--split", required=True, help="Original test split CSV (to get image paths)")
    parser.add_argument("--output", required=True, help="Path to write the review CSV")
    parser.add_argument("--sample", type=int, default=25, help="Number of rows to sample")
    args = parser.parse_args()

    # Load generated JSON (list ordered same as split CSV).
    generated = json.load(Path(args.generated).open(encoding="utf-8"))
    # Load split CSV to get image paths.
    split_df = pd.read_csv(args.split)
    if len(split_df) != len(generated):
        raise ValueError("Generated JSON length does not match split CSV length")

    # Combine into a single DataFrame.
    combined = pd.DataFrame({
        "image_path": split_df["image_path"],
        "reference_report": split_df["report_text"],
        "generated_report": [entry["generated_report"] for entry in generated],
    })

    # Randomly sample without replacement.
    sample_df = combined.sample(n=min(args.sample, len(combined)), random_state=42)
    # Remove any personally identifying columns – we already only have image_path.
    sample_df.to_csv(args.output, index=False)
    print(f"Wrote {len(sample_df)} rows to {args.output}")

if __name__ == "__main__":
    main()
