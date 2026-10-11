from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .answering import answer_question
from .chunking import fixed_size_chunks, structure_aware_chunks
from .evaluation import evaluate
from .models import EvaluationCase, Page
from .retrieval import Embedder, PolicyIndex


def compare_chunking(pages: list[Page], cases: list[EvaluationCase], embedder: Embedder) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, chunks in (
        ("fixed", fixed_size_chunks(pages)),
        ("structure", structure_aware_chunks(pages)),
    ):
        index = PolicyIndex(chunks, embedder)
        frame = evaluate(index, cases, "semantic", threshold=0.0)
        frame["chunking"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def threshold_sweep(
    index: PolicyIndex,
    cases: list[EvaluationCase],
    method: str,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    values = thresholds or [round(value * 0.025, 3) for value in range(0, 25)]
    rows: list[dict] = []
    for threshold in values:
        decisions = [
            (case, answer_question(index, case.question, method=method, threshold=threshold))
            for case in cases
        ]
        answerable = [(case, answer) for case, answer in decisions if case.answerable]
        unanswerable = [(case, answer) for case, answer in decisions if not case.answerable]
        rows.append(
            {
                "threshold": threshold,
                "correct_refusal_rate": sum(answer.status == "refused" for _, answer in unanswerable) / max(len(unanswerable), 1),
                "false_refusal_rate": sum(answer.status == "refused" for _, answer in answerable) / max(len(answerable), 1),
                "answerable_count": len(answerable),
                "unanswerable_count": len(unanswerable),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame, false_refusal_weight: float = 1.5) -> float:
    """Choose on development data only; penalize rejecting valid questions."""
    scored = curve.copy()
    scored["utility"] = scored["correct_refusal_rate"] - false_refusal_weight * scored["false_refusal_rate"]
    best = scored.sort_values(["utility", "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"])

