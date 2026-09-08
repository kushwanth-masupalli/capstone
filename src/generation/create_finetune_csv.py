#!/usr/bin/env python3
"""Create reproducible patient-level splits for report-generation training.

The generated files are data/report_splits/train.csv,
data/report_splits/validation.csv, and data/report_splits/test.csv.  No
patient (uid) occurs in more than one split.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
DATA_DIR = Path("data")
SPLIT_DIR = DATA_DIR / "report_splits"


def main() -> None:
    reports = pd.read_csv(DATA_DIR / "indiana_reports.csv")
    projections = pd.read_csv(DATA_DIR / "indiana_projections.csv")

    frontal = projections.loc[projections["projection"] == "Frontal"].copy()
    merged = frontal.merge(reports[["uid", "findings"]], on="uid", how="inner")
    merged = merged.rename(columns={"findings": "report_text"})
    merged["report_text"] = merged["report_text"].fillna("").astype(str).str.strip()
    merged["image_path"] = merged["filename"].map(
        lambda filename: str(DATA_DIR / "images" / "preprocessed" / filename)
    )
    merged["sample_id"] = merged["filename"].astype(str)

    usable = merged.loc[merged["report_text"].ne("")].copy()
    usable = usable.loc[usable["image_path"].map(lambda path: Path(path).is_file())]
    usable = usable[["sample_id", "uid", "image_path", "report_text"]].reset_index(drop=True)

    uids = usable["uid"].drop_duplicates().to_numpy()
    train_validation_uids, test_uids = train_test_split(
        uids, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_uids, validation_uids = train_test_split(
        train_validation_uids,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
    )
    split_uids = {
        "train": set(train_uids),
        "validation": set(validation_uids),
        "test": set(test_uids),
    }
    assert split_uids["train"].isdisjoint(split_uids["validation"])
    assert split_uids["train"].isdisjoint(split_uids["test"])
    assert split_uids["validation"].isdisjoint(split_uids["test"])

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in split_uids.items():
        split = usable.loc[usable["uid"].isin(ids)].reset_index(drop=True)
        split.to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        print(f"{name:10s}: {len(split):4d} images / {split['uid'].nunique():4d} patients")

    print(f"Saved patient-level report splits to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
