from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .answering import answer_question
from .chunking import fixed_size_chunks, structure_aware_chunks
from .documents import extract_pages
from .models import EvaluationCase, EvaluationResult, RetrievedChunk
from .retrieval import HashingEmbedder, PolicyIndex, SentenceTransformerEmbedder, tokenize
from .failures import failure_summary
from .grading import build_manual_grading_rows
from .dataset import validate_cases
from .reporting import write_markdown_report


def load_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cases.append(
                EvaluationCase(
                    id=row["id"],
                    split=row["split"],  # type: ignore[arg-type]
                    question=row["question"],
                    expected_answer=row["expected_answer"],
                    relevant_documents=tuple(filter(None, row["relevant_documents"].split("|"))),
                    relevant_pages=tuple(int(page) for page in row["relevant_pages"].split("|") if page),
                    question_type=row["question_type"],
                    answerable=row["answerable"].strip().lower() == "true",
                )
            )
    return cases


def correct_rank(case: EvaluationCase, retrieved: list[RetrievedChunk]) -> int | None:
    if not case.answerable:
        return None
    for item in retrieved:
        document_matches = not case.relevant_documents or item.chunk.document in case.relevant_documents
        page_matches = not case.relevant_pages or bool(set(item.chunk.pages) & set(case.relevant_pages))
        if document_matches and page_matches:
            return item.rank
    return None


def answer_is_faithful(answer: str, retrieved: list[RetrievedChunk]) -> bool | None:
    """Conservative lexical audit for the offline extractive mode.

    It is not marketed as semantic faithfulness. A production LLM answer should be
    graded by a documented LLM rubric and manually spot-checked.
    """
    if not answer or not retrieved:
        return None
    evidence_terms = set(tokenize(" ".join(item.chunk.text for item in retrieved)))
    answer_terms = {term for term in tokenize(answer) if len(term) > 3}
    return bool(answer_terms) and len(answer_terms & evidence_terms) / len(answer_terms) >= 0.85


def evaluate(index: PolicyIndex, cases: list[EvaluationCase], configuration: str, threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    for case in cases:
        answer = answer_question(index, case.question, method=configuration, threshold=threshold)
        retrieved = list(answer.evidence)
        rank = correct_rank(case, retrieved)
        result = EvaluationResult(
            case_id=case.id,
            configuration=configuration,
            split=case.split,
            answerable=case.answerable,
            correct_rank=rank,
            hit_at_1=rank == 1,
            hit_at_3=rank is not None and rank <= 3,
            hit_at_5=rank is not None and rank <= 5,
            reciprocal_rank=1 / rank if rank else 0,
            refused=answer.status == "refused",
            correct_refusal=not case.answerable and answer.status == "refused",
            false_refusal=case.answerable and answer.status == "refused",
            faithful=answer_is_faithful(answer.text, retrieved) if answer.status == "answered" else None,
            latency_ms=answer.latency_ms,
            estimated_cost_usd=answer.estimated_cost_usd,
            generated_answer=answer.text,
            retrieved=[
                {
                    "rank": item.rank,
                    "document": item.chunk.document,
                    "pages": list(item.chunk.pages),
                    "score": item.score,
                    "chunk_id": item.chunk.id,
                }
                for item in retrieved
            ],
        )
        row = asdict(result)
        row["question"] = case.question
        row["question_type"] = case.question_type
        row["expected_answer"] = case.expected_answer
        row["retrieved"] = json.dumps(row["retrieved"])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_interval(values: pd.Series, iterations: int = 2000, seed: int = 7) -> tuple[float, float]:
    data = values.astype(float).to_numpy()
    if not len(data):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(iterations, len(data)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict] = []
    for (configuration, split), group in results.groupby(["configuration", "split"]):
        answerable = group[group["answerable"]]
        unanswerable = group[~group["answerable"]]
        hit_low, hit_high = bootstrap_interval(answerable["hit_at_3"])
        summaries.append(
            {
                "configuration": configuration,
                "split": split,
                "questions": len(group),
                "hit@1_count": int(answerable["hit_at_1"].sum()),
                "hit@1": answerable["hit_at_1"].mean(),
                "hit@3_count": int(answerable["hit_at_3"].sum()),
                "hit@3": answerable["hit_at_3"].mean(),
                "hit@3_ci_low": hit_low,
                "hit@3_ci_high": hit_high,
                "hit@5": answerable["hit_at_5"].mean(),
                "mrr": answerable["reciprocal_rank"].mean(),
                "correct_refusal_rate": unanswerable["correct_refusal"].mean(),
                "false_refusal_rate": answerable["false_refusal"].mean(),
                "faithfulness_rate": group["faithful"].dropna().mean(),
                "latency_ms": group["latency_ms"].mean(),
                "cost_per_query_usd": group["estimated_cost_usd"].mean(),
            }
        )
    return pd.DataFrame(summaries)


def build_index(document_paths: list[str], chunking: str, embedder_name: str = "hashing") -> PolicyIndex:
    pages = [page for path in document_paths for page in extract_pages(path)]
    chunks = fixed_size_chunks(pages) if chunking == "fixed" else structure_aware_chunks(pages)
    embedder = SentenceTransformerEmbedder() if embedder_name == "sentence-transformer" else HashingEmbedder()
    return PolicyIndex(chunks, embedder)


def calibrate_threshold(index: PolicyIndex, cases: list[EvaluationCase], method: str) -> float:
    """Choose a method-specific threshold using development cases only."""
    development = [case for case in cases if case.split == "development"]
    scored = [
        (case, (results[0].score if (results := index.search(case.question, method=method, top_k=1)) else 0.0))
        for case in development
    ]
    answerable = [score for case, score in scored if case.answerable]
    unanswerable = [score for case, score in scored if not case.answerable]
    candidates = np.linspace(0, 1, 101)
    best_threshold = 0.0
    best_utility = float("-inf")
    for threshold in candidates:
        false_refusals = sum(score < threshold for score in answerable) / max(len(answerable), 1)
        correct_refusals = sum(score < threshold for score in unanswerable) / max(len(unanswerable), 1)
        utility = correct_refusals - 1.5 * false_refusals
        if utility > best_utility or (utility == best_utility and threshold > best_threshold):
            best_utility = utility
            best_threshold = float(threshold)
    return best_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PolicyLens retrieval configurations")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--chunking", choices=["fixed", "structure"], default="structure")
    parser.add_argument("--embedder", choices=["hashing", "sentence-transformer"], default="hashing")
    parser.add_argument("--threshold", default="auto", help="A number or 'auto' for development-set calibration")
    parser.add_argument("--output", default="artifacts/evaluation_results.csv")
    args = parser.parse_args()

    index = build_index(args.documents, args.chunking, args.embedder)
    cases = load_cases(args.cases)
    issues = validate_cases(cases)
    errors = [issue for issue in issues if issue.level == "error"]
    for issue in issues:
        print(f"[{issue.level.upper()}] {issue.message}")
    if errors:
        raise SystemExit("Evaluation dataset validation failed.")
    methods = ("semantic", "hybrid", "hybrid_rerank")
    thresholds = (
        {method: calibrate_threshold(index, cases, method) for method in methods}
        if args.threshold == "auto"
        else {method: float(args.threshold) for method in methods}
    )
    print("Thresholds selected on development data:", thresholds)
    frames = []
    for method in methods:
        frame = evaluate(index, cases, method, thresholds[method])
        frame["threshold"] = thresholds[method]
        frame["embedder"] = index.embedder.name
        frame["chunking"] = args.chunking
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary = summarize(results)
    summary.to_csv(output.with_name("evaluation_summary.csv"), index=False)
    build_manual_grading_rows(results).to_csv(output.with_name("manual_grading_template.csv"), index=False)
    failure_summary(results).to_csv(output.with_name("failure_summary.csv"), index=False)
    write_markdown_report(summary, results, output.with_name("evaluation_report.md"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
