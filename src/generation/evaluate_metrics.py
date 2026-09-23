#!/usr/bin/env python3
"""Evaluate generated reports against references using pycocoevalcap.

Usage:
    python src/generation/evaluate_metrics.py \
        results/finetuned_patient_split/pipeline_results.json \
        data/report_splits/test.csv \
        --output results/finetuned_patient_split/evaluation.json

The script loads the JSON produced by `generate_reports.py` (list of dicts with
`reference_report` and `generated_report`), builds the structures required by
`pycocoevalcap`, runs the evaluation, and writes a compact JSON with
BLEU-1, ROUGE-L, METEOR and CIDEr scores.
"""

import argparse
import json
import re
from pathlib import Path

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu


# Matches a full chat-template transcript leaked into a "report": the
# system role line, the user role + instruction, and the assistant marker.
# Produced by older decoding that did not slice off the prompt tokens.
_CHAT_TEMPLATE_RE = re.compile(
    r"^\s*system\s*\n.*?\buser\s*\n.*?\bassistant\s*\n?",
    flags=re.DOTALL | re.IGNORECASE,
)


def strip_chat_template_leakage(text: str) -> str:
    """Remove chat-template boilerplate leaked into a generated report.

    Defensive cleanup for result JSONs produced before the decode-trimming fix.
    Clean artifacts pass through unchanged (no false edits).
    """
    return _CHAT_TEMPLATE_RE.sub("", str(text)).strip()


def clean(text):
    """Collapse all whitespace/newlines into single spaces.

    This is required because METEOR talks to a Java subprocess over a
    line-based (one message per line) protocol. Any embedded newline in a
    hypothesis or reference string desyncs the read/write counts and causes
    an infinite hang. Collapsing whitespace also makes BLEU/ROUGE/CIDEr
    tokenization more consistent.
    """
    return " ".join(str(text).split())


def build_coco_format(results):
    """Convert the list of dicts into pycocoevalcap format."""

    refs = {}
    hyps = {}

    for idx, entry in enumerate(results):
        img_id = str(idx)

        refs[img_id] = [clean(entry["reference_report"])]
        hyps[img_id] = [clean(strip_chat_template_leakage(entry["generated_report"]))]

    return refs, hyps


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated reports."
    )

    parser.add_argument(
        "generated_json",
        help="Path to JSON with generated results."
    )

    parser.add_argument(
        "split_csv",
        help="Path to the original split CSV (for sanity check)."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to write evaluation JSON."
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load generated results
    # ---------------------------------------------------------

    generated_path = Path(args.generated_json)

    with generated_path.open(
        encoding="utf-8"
    ) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} generated reports.")

    # ---------------------------------------------------------
    # Build metric input structures
    # ---------------------------------------------------------

    refs, hyps = build_coco_format(results)

    print(f"Prepared {len(refs)} report pairs for evaluation.")

    # ---------------------------------------------------------
    # Prepare evaluators
    # ---------------------------------------------------------

    bleu = Bleu(4)
    rouge = Rouge()
    meteor = Meteor()
    cider = Cider()

    # ---------------------------------------------------------
    # Run metrics
    # ---------------------------------------------------------

    print("Calculating BLEU...")

    bleu_score, _ = bleu.compute_score(
        refs,
        hyps
    )

    print("Calculating ROUGE-L...")

    rouge_score, _ = rouge.compute_score(
        refs,
        hyps
    )

    print("Calculating METEOR...")

    meteor_score, _ = meteor.compute_score(
        refs,
        hyps
    )

    print("Calculating CIDEr...")

    cider_score, _ = cider.compute_score(
        refs,
        hyps
    )

    # ---------------------------------------------------------
    # Store results
    # ---------------------------------------------------------

    # BLEU returns four scores:
    # BLEU-1, BLEU-2, BLEU-3, BLEU-4
    #
    # We extract BLEU-1 as requested.

    eval_dict = {
        "BLEU-1": float(bleu_score[0]),
        "ROUGE-L": float(rouge_score),
        "METEOR": float(meteor_score),
        "CIDEr": float(cider_score),
    }

    # ---------------------------------------------------------
    # Write output
    # ---------------------------------------------------------

    out_path = Path(args.output)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path.write_text(
        json.dumps(
            eval_dict,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("========================================")
    print("Evaluation Results")
    print("========================================")
    print(f"BLEU-1 : {eval_dict['BLEU-1']:.4f}")
    print(f"ROUGE-L: {eval_dict['ROUGE-L']:.4f}")
    print(f"METEOR : {eval_dict['METEOR']:.4f}")
    print(f"CIDEr  : {eval_dict['CIDEr']:.4f}")
    print("========================================")
    print()
    print(f"Evaluation written to: {out_path}")


if __name__ == "__main__":
    main()