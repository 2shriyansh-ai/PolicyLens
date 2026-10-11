from __future__ import annotations

from pathlib import Path

import pandas as pd

from .failures import failure_summary


def _percentage(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):.1%}"


def write_markdown_report(summary: pd.DataFrame, results: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    failures = failure_summary(results)
    lines = [
        "# PolicyLens evaluation report",
        "",
        "> Generated from the labelled evaluation set. Small differences are suggestive, not proof of universal superiority.",
        "",
        "## Retrieval and refusal results",
        "",
        "| Configuration | Split | Questions | Hit@1 | Hit@3 | 95% bootstrap CI | MRR | Correct refusal | False refusal | Latency | Cost/query |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        interval = f"{_percentage(row['hit@3_ci_low'])}–{_percentage(row['hit@3_ci_high'])}"
        lines.append(
            "| {configuration} | {split} | {questions} | {hit1} | {hit3} | {interval} | {mrr:.3f} | {correct_refusal} | {false_refusal} | {latency:.1f} ms | ${cost:.6f} |".format(
                configuration=row["configuration"],
                split=row["split"],
                questions=int(row["questions"]),
                hit1=_percentage(row["hit@1"]),
                hit3=_percentage(row["hit@3"]),
                interval=interval,
                mrr=float(row["mrr"]),
                correct_refusal=_percentage(row["correct_refusal_rate"]),
                false_refusal=_percentage(row["false_refusal_rate"]),
                latency=float(row["latency_ms"]),
                cost=float(row["cost_per_query_usd"]),
            )
        )

    lines.extend(["", "## Failure analysis", ""])
    if failures.empty:
        lines.append("No automatically classified failures were recorded.")
    else:
        lines.extend(
            [
                "| Configuration | Question type | Failure category | Count |",
                "|---|---|---|---:|",
            ]
        )
        for _, row in failures.iterrows():
            lines.append(
                f"| {row['configuration']} | {row['question_type']} | {row['failure_category']} | {int(row['count'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Report raw counts with percentages.",
            "- Select thresholds and configurations on development data only.",
            "- Use held-out results for final claims.",
            "- Manually review answer correctness and a sample of faithfulness grades.",
            "- Do not describe small or overlapping differences as proven improvements.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination

