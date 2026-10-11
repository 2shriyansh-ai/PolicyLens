from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .models import EvaluationCase


@dataclass(frozen=True)
class DatasetIssue:
    level: str
    message: str


def validate_cases(cases: list[EvaluationCase]) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    ids = [case.id for case in cases]
    duplicates = [identifier for identifier, count in Counter(ids).items() if count > 1]
    if duplicates:
        issues.append(DatasetIssue("error", f"Duplicate case IDs: {', '.join(duplicates)}"))

    normalized_questions = [" ".join(case.question.lower().split()) for case in cases]
    duplicate_questions = [question for question, count in Counter(normalized_questions).items() if count > 1]
    if duplicate_questions:
        issues.append(DatasetIssue("warning", "Exact question text appears in more than one case."))

    splits = Counter(case.split for case in cases)
    if not splits.get("development") or not splits.get("test"):
        issues.append(DatasetIssue("error", "Both development and test splits are required."))
    elif splits["test"] / len(cases) < 0.25:
        issues.append(DatasetIssue("warning", "Held-out test split is below 25% of the dataset."))

    if sum(not case.answerable for case in cases) < 2:
        issues.append(DatasetIssue("warning", "Use multiple unanswerable questions to tune refusal safely."))

    for case in cases:
        if case.answerable and (not case.relevant_documents or not case.relevant_pages):
            issues.append(DatasetIssue("error", f"{case.id}: answerable case lacks labelled evidence."))
        if not case.answerable and (case.relevant_documents or case.relevant_pages):
            issues.append(DatasetIssue("warning", f"{case.id}: unanswerable case has evidence labels."))
        if not case.question.strip() or not case.expected_answer.strip():
            issues.append(DatasetIssue("error", f"{case.id}: question and expected answer are required."))

    category_counts = Counter(case.question_type for case in cases)
    for category, count in category_counts.items():
        if count < 10:
            issues.append(
                DatasetIssue(
                    "info",
                    f"Category '{category}' has only {count} case(s); treat category percentages as descriptive.",
                )
            )
    return issues

