#!/usr/bin/env python3
"""Evaluate generated reports against references using pycocoevalcap.

Usage:
    python src/generation/evaluate_metrics.py \
        results/finetuned_patient_split/pipeline_results.json \
        data/report_splits/test.csv \
        --output results/finetuned_patient_split/evaluation.json

The script loads the JSON produced by `generate_reports.py` (list of dicts with
`reference_report` and `generated_report`), builds the COCO‑style structures
required by `pycocoevalcap`, runs the evaluation, and writes a compact JSON with
BLEU‑1, ROUGE‑L, METEOR and CIDEr scores.
"""

import argparse
import json
from pathlib import Path

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu

def build_coco_format(results):
    """Convert the list of dicts into two COCO‑style JSON structures.

    Returns:
        refs: dict mapping image_id -> list of reference strings
        hyps: dict mapping image_id -> generated string
    """
    refs = {}
    hyps = {}
    for idx, entry in enumerate(results):
        img_id = str(idx)  # use index as a synthetic ID
        refs[img_id] = [entry["reference_report"]]
        hyps[img_id] = entry["generated_report"]
    return refs, hyps

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated reports.")
    parser.add_argument("generated_json", help="Path to JSON with generated results.")
    parser.add_argument("split_csv", help="Path to the original split CSV (for sanity check).")
    parser.add_argument("--output", required=True, help="Path to write evaluation JSON.")
    args = parser.parse_args()

    # Load generated results.
    generated_path = Path(args.generated_json)
    results = json.load(generated_path.open(encoding="utf-8"))

    # Build COCO structures.
    refs, hyps = build_coco_format(results)

    # Prepare evaluators.
    bleu = Bleu(4)  # we will extract BLEU‑1 later
    rouge = Rouge()
    meteor = Meteor()
    cider = Cider()

    # The COCOEvalCap wrapper expects a ground‑truth object with a specific API.
    # We'll create minimal mock objects.
    class MockCoco:
        def __init__(self, data):
            self.data = data
        def getImgIds(self):
            return list(self.data.keys())
        def getCapIds(self, imgIds=None):
            # Return dummy caption IDs (same as image IDs).
            return list(self.data.keys())
        def loadAnns(self, ids):
            # Return list of dicts each with 'caption'
            return [{"caption": self.data[i][0]} for i in ids]
    class MockCocoRes:
        def __init__(self, data):
            self.data = data
        def getImgIds(self):
            return list(self.data.keys())
        def getCaptions(self, imgIds=None):
            return {i: self.data[i] for i in (imgIds or self.getImgIds())}

    coco = MockCoco(refs)
    coco_res = MockCocoRes(hyps)

    # Run each metric individually.
    bleu_score, _ = bleu.compute_score(coco, coco_res)
    rouge_score, _ = rouge.compute_score(coco, coco_res)
    meteor_score, _ = meteor.compute_score(coco, coco_res)
    cider_score, _ = cider.compute_score(coco, coco_res)

    # bleu returns a list of 4 scores (BLEU‑1..4). Use the first.
    eval_dict = {
        "BLEU-1": float(bleu_score[0]),
        "ROUGE-L": float(rouge_score),
        "METEOR": float(meteor_score),
        "CIDEr": float(cider_score),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eval_dict, indent=2))
    print(f"Evaluation written to {out_path}")

if __name__ == "__main__":
    main()
