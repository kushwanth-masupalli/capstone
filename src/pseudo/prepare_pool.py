"""
Phase 1 — Build an unlabeled real chest X-ray pool (NIH ChestX-ray14)
======================================================================
Turns a folder of RAW NIH ChestX-ray14 images into `data/pseudo_pool/`:

    data/pseudo_pool/PP_00000001.png ...  224x224 PNGs, processed with the
                                          SAME CLAHE -> resize -> RGB pipeline
                                          used for IU (preprocess.py), so the
                                          teacher sees the pool images in the
                                          same distribution it was trained on.
    data/pseudo_pool/pool_manifest.csv    provenance per image (source file,
                                          NIH view position / finding labels)

NIH's own 14 labels are deliberately IGNORED downstream — the pool is treated
as unlabeled. The manifest keeps them only for provenance/analysis.

Getting the raw images (no registration required):
    1. https://nihcc.app.box.com/v/ChestXray-NIHCC  (official NIH release)
       Download any/all images_*.tar.gz archives and extract them — each
       archive is an independent subset, so you can stop after enough images.
    2. Extract to e.g. data/nih_raw/ so each archive becomes a subfolder of
       PNGs (the filenames are globally unique across archives).

Usage:
    python -m src.pseudo.prepare_pool \
        --source data/nih_raw \
        --data_entry_csv data/nih_raw/Data_Entry_2017.csv \
        --max_images 20000

Outputs a deterministic random sample (--seed) of the requested size — you
don't need the full 112k-image dataset (15-20k candidates is the plan's
target, since the confidence filter in Phase 3 discards a lot).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess import preprocess_image  # same CLAHE -> 224x224 -> RGB pipeline as IU

POOL_DIR = Path("data/pseudo_pool")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
KEEP_VIEWS = {"PA", "AP"}  # frontal only — matches the IU frontal domain


def find_images(source: Path):
    """Recursively list candidate image files under `source` (sorted for determinism)."""
    imgs = []
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    return imgs


def filter_frontal_by_csv(image_paths, data_entry_csv):
    """
    If the NIH Data_Entry_2017.csv is available, drop lateral views so the
    pool matches the frontal-only IU domain. Returns (kept, dropped_count).
    """
    df = pd.read_csv(data_entry_csv)
    if "Image Index" not in df.columns or "View Position" not in df.columns:
        print(f"WARNING: {data_entry_csv} lacks 'Image Index'/'View Position' — "
              f"skipping view filter (laterals may enter the pool).")
        return image_paths, 0
    view = df.set_index("Image Index")["View Position"].to_dict()
    kept, dropped = [], 0
    for p in image_paths:
        v = view.get(p.name)
        if v in KEEP_VIEWS or v is None:
            kept.append(p)  # None = unknown; keep (better than silently dropping)
        else:
            dropped += 1
    print(f"  View filter ({'/'.join(sorted(KEEP_VIEWS))} kept): dropped {dropped} non-frontal")
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description="Build the unlabeled NIH pseudo-label pool")
    parser.add_argument("--source", type=str, required=True,
                        help="Folder of raw NIH ChestX-ray14 PNGs (recursively scanned)")
    parser.add_argument("--data_entry_csv", type=str, default="",
                        help="Optional NIH Data_Entry_2017.csv for view-position filtering")
    parser.add_argument("--pool_dir", type=str, default=str(POOL_DIR),
                        help="Where to write the resized pool (default: data/pseudo_pool)")
    parser.add_argument("--max_images", type=int, default=20000,
                        help="How many candidate images to keep (default: 20000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"ERROR: source folder not found: {source}")
        sys.exit(1)

    pool_dir = Path(args.pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Enumerate + optionally filter by view position ──
    images = find_images(source)
    print(f"Found {len(images)} raw images under {source}")
    if not images:
        print("Nothing to do. Point --source at extracted NIH ChestX-ray14 PNGs.")
        sys.exit(1)

    if args.data_entry_csv:
        images, _ = filter_frontal_by_csv(images, Path(args.data_entry_csv))
        if not images:
            print("No images left after view filtering.")
            sys.exit(1)

    # ── 2. Deterministic subsample ──
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(images))
    images = [images[i] for i in order[: args.max_images]]
    print(f"Keeping {len(images)} candidate images (seed={args.seed}, max={args.max_images})")

    # ── 3. Preprocess each (CLAHE -> 224x224 RGB PNG), resumable ──
    # WARNING: resume assumes you re-run with the SAME source/seed/max_images,
    # otherwise the PP_* index <-> source mapping shifts and must be redone
    # from an empty pool dir (delete data/pseudo_pool/PP_*.png to restart).
    existing = sorted(pool_dir.glob("PP_*.png"))
    start_idx = len(existing)
    print(f"Pool already contains {start_idx} images - resuming from there.\n")

    success, failed = 0, 0
    manifest_rows = []
    for i, src_img in enumerate(images, start=start_idx + 1):
        dst = pool_dir / f"PP_{i:08d}.png"
        if dst.exists():
            success += 1
        elif preprocess_image(src_img, dst):
            success += 1
        else:
            failed += 1
            print(f"  [SKIP] could not load {src_img.name}")
            continue
        manifest_rows.append({
            "pool_filename": dst.name,
            "source_filename": src_img.name,
            "sample_index": i,
        })
        if i % 500 == 0:
            print(f"  processed {i}/{start_idx + len(images)}")

    # ── 4. Provenance manifest (merges with any earlier runs, dedup by file) ──
    manifest_path = pool_dir / "pool_manifest.csv"
    merged = {}
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                merged[row["pool_filename"]] = row
    for row in manifest_rows:
        merged[row["pool_filename"]] = row
    rows = sorted(merged.values(), key=lambda r: int(r["sample_index"]))
    if rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print("\n" + "=" * 60)
    print(f"Pool ready: {pool_dir}  ({success} images written/kept, {failed} failed)")
    print(f"Manifest:   {manifest_path}")
    print(f"Expected pool size: ~{success}  (Phase 3 will filter this down)")
    print("\nNext step:")
    print("  python -m src.pseudo.generate_pseudo_labels")


if __name__ == "__main__":
    main()
