from __future__ import annotations

import json

import pandas as pd


def classify_failure(row: pd.Series) -> str:
    if row.get("false_refusal", False):
        return "false_refusal"
    if not row.get("answerable", True):
        return "none" if row.get("refused", False) else "unsafe_answer"
    rank = row.get("correct_rank")
    if pd.isna(rank):
        question_type = row.get("question_type", "")
        if question_type == "table":
            return "table_extraction_or_retrieval"
        if question_type == "multi_page":
            return "incomplete_multi_passage_retrieval"
        return "retrieval_miss"
    if int(rank) > 1:
        return "ranking_failure"
    if row.get("faithful") is False:
        return "generation_or_citation_failure"
    return "none"


def add_failure_categories(results: pd.DataFrame) -> pd.DataFrame:
    enriched = results.copy()
    enriched["failure_category"] = enriched.apply(classify_failure, axis=1)
    return enriched


def failure_summary(results: pd.DataFrame) -> pd.DataFrame:
    enriched = add_failure_categories(results)
    failures = enriched[enriched["failure_category"] != "none"]
    if failures.empty:
        return pd.DataFrame(columns=["configuration", "question_type", "failure_category", "count"])
    return (
        failures.groupby(["configuration", "question_type", "failure_category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
