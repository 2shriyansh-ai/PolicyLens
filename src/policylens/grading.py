from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from .config import LLMSettings
from .retrieval import tokenize


@dataclass(frozen=True)
class AnswerGrade:
    score: int
    label: str
    explanation: str
    method: str


def lexical_reference_grade(expected: str, generated: str) -> AnswerGrade:
    """Transparent offline proxy, never presented as human answer accuracy."""
    expected_terms = {token for token in tokenize(expected) if len(token) > 2}
    generated_terms = set(tokenize(generated))
    coverage = len(expected_terms & generated_terms) / max(len(expected_terms), 1)
    if coverage >= 0.8:
        return AnswerGrade(2, "correct", f"Reference-term coverage {coverage:.0%}", "lexical-proxy")
    if coverage >= 0.4:
        return AnswerGrade(1, "partial", f"Reference-term coverage {coverage:.0%}", "lexical-proxy")
    return AnswerGrade(0, "incorrect", f"Reference-term coverage {coverage:.0%}", "lexical-proxy")


def build_manual_grading_rows(results):
    columns = ["case_id", "question", "expected_answer", "generated_answer", "configuration", "split"]
    frame = results[columns].copy()
    frame["manual_score_0_1_2"] = ""
    frame["manual_faithful_yes_no"] = ""
    frame["reviewer_notes"] = ""
    return frame


JUDGE_PROMPT = """Grade a policy answer against a reference and evidence.
Return JSON only with: correctness_score (0, 1, or 2), faithful (boolean), explanation.
2 means fully correct, 1 means partially correct or incomplete, 0 means incorrect.
Faithful means every material claim is supported by the evidence.
"""


def llm_grade(settings: LLMSettings, question: str, expected: str, generated: str, evidence: str) -> AnswerGrade:
    response = requests.post(
        f"{settings.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"},
        json={
            "model": settings.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": f"QUESTION: {question}\nREFERENCE: {expected}\nANSWER: {generated}\nEVIDENCE: {evidence}",
                },
            ],
        },
        timeout=settings.timeout_seconds,
    )
    response.raise_for_status()
    payload = json.loads(response.json()["choices"][0]["message"]["content"])
    score = int(payload["correctness_score"])
    return AnswerGrade(score, {0: "incorrect", 1: "partial", 2: "correct"}[score], payload["explanation"], "llm-judge")
