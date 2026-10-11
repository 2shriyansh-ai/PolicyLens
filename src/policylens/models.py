from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Page:
    document: str
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    pages: tuple[int, ...]
    text: str
    heading: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    semantic_score: float | None = None
    keyword_score: float | None = None


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    status: Literal["answered", "refused"]
    evidence: tuple[RetrievedChunk, ...]
    confidence_score: float
    latency_ms: float
    reason: str | None = None
    generator: str = "extractive"
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    split: Literal["development", "test"]
    question: str
    expected_answer: str
    relevant_documents: tuple[str, ...]
    relevant_pages: tuple[int, ...]
    question_type: str
    answerable: bool


@dataclass
class EvaluationResult:
    case_id: str
    configuration: str
    split: str
    answerable: bool
    correct_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    refused: bool
    correct_refusal: bool
    false_refusal: bool
    faithful: bool | None
    latency_ms: float
    estimated_cost_usd: float
    generated_answer: str
    retrieved: list[dict] = field(default_factory=list)
