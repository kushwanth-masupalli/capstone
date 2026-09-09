import re
from typing import List, Set

def extract_findings(report_text: str, vocab: List[str]) -> Set[str]:
    """Extract mentioned findings from a report using simple keyword matching.

    The function lower‑cases both the report and the vocabulary entries and
    performs a word‑boundary search. A small synonym handling could be added
    later, but for the demo the exact label match is sufficient.
    """
    text = report_text.lower()
    found = set()
    for label in vocab:
        # Normalise the label for matching – replace hyphens/underscores with spaces.
        norm_label = label.lower().replace("-", " ").replace("_", " ")
        # Build a word‑boundary regex to avoid partial matches.
        pattern = r"\b" + re.escape(norm_label) + r"\b"
        if re.search(pattern, text):
            found.add(label)
    return found

def compare_hallucination(predicted: Set[str], mentioned: Set[str]) -> List[str]:
    """Return human‑readable messages for findings mentioned but not predicted.

    Parameters
    ----------
    predicted: set of labels that the classifier predicted (probability ≥ 0.5).
    mentioned: set of labels extracted from the generated report.
    """
    hallucinations = mentioned - predicted
    messages = [f"Reported finding \"{f}\" not predicted by classifier" for f in sorted(hallucinations)]
    return messages
