from __future__ import annotations

import re
from typing import Optional


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute Rouge-L F1 score for a single prediction/reference pair."""
    from rouge_score import rouge_scorer as _rs

    scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


_SST2_RE = re.compile(r"\b(positive|negative)\b", re.IGNORECASE)


def extract_sst2_label(text: str) -> str:
    m = _SST2_RE.search(text or "")
    if not m:
        return ""
    return m.group(1).lower()


def normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def sciq_answer_in_text(text: str, canonical_answer: str) -> bool:
    """Check if canonical answer appears in prediction text (with light normalization)."""
    cand = normalize_text(text)
    ans = normalize_text(canonical_answer)
    if not cand or not ans:
        return False
    if ans in cand:
        return True
    # Also try punctuation-light matching.
    cand2 = re.sub(r"[^a-z0-9\s]", " ", cand)
    ans2 = re.sub(r"[^a-z0-9\s]", " ", ans)
    cand2 = re.sub(r"\s+", " ", cand2).strip()
    ans2 = re.sub(r"\s+", " ", ans2).strip()
    if not cand2 or not ans2:
        return False
    if ans2 in cand2:
        return True
    # Word-boundary check for single token answers.
    if " " not in ans2:
        return re.search(rf"\b{re.escape(ans2)}\b", cand2) is not None
    return False


def pick_canonical_sciq_answer(example: dict) -> str:
    """Retrieve canonical SciQ answer from row if present, else fallback to output."""
    v: Optional[str] = example.get("output_canonical")
    if v:
        return str(v)
    return str(example.get("output", ""))
