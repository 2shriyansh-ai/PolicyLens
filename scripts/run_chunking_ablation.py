from __future__ import annotations

from pathlib import Path

import pandas as pd

from policylens.documents import extract_pages
from policylens.evaluation import load_cases
from policylens.experiments import compare_chunking
from policylens.retrieval import SentenceTransformerEmbedder


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = sorted((ROOT / "data" / "benchmark_policies").glob("*.pdf"))
    pages = [page for path in paths for page in extract_pages(path)]
    cases = load_cases(ROOT / "data" / "benchmark_cases.csv")
    results = compare_chunking(pages, cases, SentenceTransformerEmbedder())
    summary = (
        results[results["answerable"]]
        .groupby(["chunking", "split"])
        .agg(
            cases=("case_id", "count"),
            hit_at_1=("hit_at_1", "mean"),
            hit_at_3=("hit_at_3", "mean"),
            hit_at_5=("hit_at_5", "mean"),
            mrr=("reciprocal_rank", "mean"),
            latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
    )
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    results.to_csv(artifact_dir / "chunking_ablation_results.csv", index=False)
    summary.to_csv(artifact_dir / "chunking_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

