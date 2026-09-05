"""Semi-supervised pseudo-labeling (self-training) pipeline.

Expands the real IU training set with high-confidence pseudo-labels produced
by an IU-trained "teacher" model on a large pool of REAL, unlabeled chest
X-rays (NIH ChestX-ray14). No generative model, no synthetic artifacts —
every added image is a real X-ray.

Modules:
    prepare_pool.py          Phase 1 — build a resized, unlabeled image pool
    generate_pseudo_labels.py Phase 3 — teacher inference + confidence filter
    evaluate_checkpoint.py   Phase 2/5 — record/test AUC-F1 on the real IU split

Non-negotiable rule (Phase 0): val/test stay 100% real IU images. The pool and
its pseudo-labels can ONLY ever enter the train partition.
"""
