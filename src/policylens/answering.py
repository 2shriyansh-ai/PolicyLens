from __future__ import annotations

import re
import time
from collections.abc import Callable

from .models import Answer, RetrievedChunk
from .llm import GenerationOutput
from .retrieval import PolicyIndex, tokenize


def evidence_is_sufficient(results: list[RetrievedChunk], threshold: float) -> bool:
    return bool(results) and results[0].score >= threshold


def extractive_answer(question: str, evidence: list[RetrievedChunk]) -> str:
    """Offline fallback that selects the most question-relevant evidence sentences."""
    question_terms = set(tokenize(question))
    candidates: list[tuple[float, str]] = []
    for result in evidence[:3]:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", result.chunk.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = set(tokenize(sentence))
            overlap = len(question_terms & terms) / max(len(question_terms), 1)
            candidates.append((overlap, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return "I couldn't find sufficient information in the provided policies."
    return candidates[0][1]


def answer_question(
    index: PolicyIndex,
    question: str,
    *,
    method: str = "hybrid_rerank",
    top_k: int = 5,
    threshold: float = 0.20,
    generator: Callable[[str, list[RetrievedChunk]], str | GenerationOutput] | None = None,
) -> Answer:
    started = time.perf_counter()
    results = index.search(question, method=method, top_k=top_k)
    if not evidence_is_sufficient(results, threshold):
        return Answer(
            question=question,
            text="I couldn't find sufficient information in the provided policies.",
            status="refused",
            evidence=tuple(results),
            confidence_score=results[0].score if results else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            reason="Top evidence score was below the configured threshold.",
        )

    generated = (generator or extractive_answer)(question, results)
    if isinstance(generated, GenerationOutput):
        answer_text = generated.text
        provider = generated.provider
        input_tokens = generated.input_tokens
        output_tokens = generated.output_tokens
        cost = generated.estimated_cost_usd
    else:
        answer_text = generated
        provider = "extractive"
        input_tokens = output_tokens = 0
        cost = 0.0
    return Answer(
        question=question,
        text=answer_text,
        status="answered",
        evidence=tuple(results[:3]),
        confidence_score=results[0].score,
        latency_ms=(time.perf_counter() - started) * 1000,
        generator=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
    )
