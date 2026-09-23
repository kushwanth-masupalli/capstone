#!/usr/bin/env python3
"""PLAN 4 Phase 4.2 — data hygiene for VLM training.

1. VLM-quality images (Instruction 8): letterbox the *original* PNGs to a
   square (aspect preserved, gray padding), resize to 896 with LANCZOS, save
   to ``data/images/vlm/``. The 224 px CLAHE folder stays untouched for the
   DenseNet classifier. Doing the resize offline removes the per-step CPU
   resize cost (S2/S4) while feeding the vision tower real detail (C5).
2. ``XXXX`` policy (Instruction 9, default: strip) applied to training
   targets and written to a new ``report_text_clean`` column; raw text is kept
   in ``report_text`` so metrics can be reported on both (C6).
3. Duplicate cap (Instruction 10): exact-duplicate reports capped at 3 copies
   in the *train* split only (val/test untouched) — 2,320 → ~1,994 rows.

Outputs ``data/report_splits/*.csv`` overwritten in place (raw columns kept;
``image_path`` now points at the VLM images, forward slashes) plus a printed
summary and ``results/plan4/data_hygiene.json``.

Usage:
    python scripts/make_vlm_data.py            # full run
    python scripts/make_vlm_data.py --dry-run  # stats only, no writes
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

VLM_SIZE = 896
VLM_DIR = Path("data/images/vlm")
SPLIT_DIR = Path("data/report_splits")
SPLITS = ["train", "validation", "test"]
XXXX_POLICY_CHOICES = ("strip", "keep")
DUP_CAP = 3

# "XXXX" stands for de-identified words/phrases. Stripping removes the token
# plus the whitespace run before it, then tidies stray punctuation/spacing.
XXXX_RE = re.compile(r"\s*\bXXXX\b\s*")


def clean_report(text: str, policy: str) -> str:
    if policy == "keep":
        return text
    out = XXXX_RE.sub(" ", str(text))
    out = re.sub(r"\s+([,.;:])", r"\1", out)   # "word , word" -> "word, word"
    out = re.sub(r"\(\s*\)", "", out)          # "()" leftovers
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.;")
    return out.strip()


def make_vlm_image(src: Path, dst: Path, size: int = VLM_SIZE) -> None:
    """Letterbox to a square (aspect preserved, dark-gray padding), LANCZOS."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (16, 16, 16))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    # optimize=False: ~4x faster; pixel-identical output, just bigger files.
    # These stay on disk (gitignored), so file size doesn't matter.
    canvas.save(dst, format="PNG")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build VLM images + clean CSV splits.")
    ap.add_argument("--policy", choices=XXXX_POLICY_CHOICES, default="strip",
                    help="XXXX token policy for training targets (Instruction 9).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats only; write nothing.")
    ap.add_argument("--skip-images", action="store_true",
                    help="Only rewrite the CSVs (images already generated).")
    args = ap.parse_args()

    stats = {"policy": args.policy, "dup_cap": DUP_CAP, "vlm_size": VLM_SIZE}
    cleaned_report = lambda t: clean_report(t, args.policy)  # noqa: E731

    for split in SPLITS:
        path = SPLIT_DIR / f"{split}.csv"
        if not path.exists():
            print(f"WARNING: {path} missing - skipped")
            continue
        df = pd.read_csv(path)

        # ---- resolve source image for every row (raw PNG preferred) ----
        def resolve_src(p: str) -> Path | None:
            pp = Path(str(p).replace("\\", "/"))
            cand = [Path("data/images") / pp.name,   # original full-res PNG
                    Path("data/images/preprocessed") / pp.name,
                    pp]
            return next((c for c in cand if c.is_file()), None)

        srcs = df["image_path"].map(resolve_src)
        missing = int(srcs.isna().sum())
        if missing:
            print(f"WARNING: {split}: {missing} rows have no resolvable image")

        df["vlm_image_path"] = [
            str(VLM_DIR / Path(p).name).replace("\\", "/") if p is not None else ""
            for p in srcs
        ]

        # ---- generate VLM images (train/val need them; test for eval) ----
        if not args.dry_run and not args.skip_images:
            done = skipped = 0
            for src, rel in zip(srcs, df["vlm_image_path"]):
                if src is None or not rel:
                    continue
                dst = Path(rel)
                if dst.is_file():
                    skipped += 1
                    continue
                make_vlm_image(src, dst)
                done += 1
            print(f"{split}: generated {done} VLM images, reused {skipped}")

        # ---- report hygiene ----
        df["report_text_clean"] = df["report_text"].map(cleaned_report)

        n_before = len(df)
        xxxx_frac_raw = float(df["report_text"].str.contains("XXXX").mean())
        xxxx_frac_clean = float(df["report_text_clean"].str.contains("XXXX").mean())
        dup_counts = df["report_text"].value_counts()
        dup_rows = int(dup_counts[dup_counts > 1].sum() - (dup_counts > 1).sum())

        if split == "train" and not args.dry_run:
            # Cap = keep at most DUP_CAP copies per exact report text: reports
            # with count <= cap are untouched, longer ones keep their FIRST 3
            # occurrences (2,320 -> ~1,994 rows, matching the plan's measurement).
            occurrence = df.groupby("report_text").cumcount()
            dup_counts = df["report_text"].value_counts()
            within_cap = dup_counts[df["report_text"]].to_numpy() <= DUP_CAP
            first_copies = occurrence.to_numpy() < DUP_CAP
            df = df[within_cap | first_copies].reset_index(drop=True)

        stats[split] = {
            "rows_before": n_before,
            "rows_after": len(df),
            "rows_dropped": n_before - len(df),
            "exact_dup_reports": dup_rows,
            "xxxx_frac_raw": round(xxxx_frac_raw, 4),
            "xxxx_frac_clean": round(xxxx_frac_clean, 4),
            "missing_images": missing,
        }
        print(f"{split}: {n_before} -> {len(df)} rows | dup reports: {dup_rows} | "
              f"XXXX frac raw={xxxx_frac_raw:.3f} clean={xxxx_frac_clean:.3f}")

        if not args.dry_run:
            df.to_csv(path, index=False)

    if not args.dry_run:
        out = Path("results/plan4/data_hygiene.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nstats saved to {out}")
    else:
        print("\n(dry run - nothing written)")


if __name__ == "__main__":
    main()
