from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    timeout_seconds: float = 45.0

    @classmethod
    def from_environment(cls) -> "LLMSettings | None":
        key = os.getenv("POLICYLENS_LLM_API_KEY", "").strip()
        if not key:
            return None
        return cls(
            base_url=os.getenv("POLICYLENS_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=key,
            model=os.getenv("POLICYLENS_LLM_MODEL", "gpt-4.1-mini"),
            input_cost_per_million=float(os.getenv("POLICYLENS_INPUT_COST_PER_MILLION", "0")),
            output_cost_per_million=float(os.getenv("POLICYLENS_OUTPUT_COST_PER_MILLION", "0")),
            timeout_seconds=float(os.getenv("POLICYLENS_LLM_TIMEOUT_SECONDS", "45")),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    base_url: str
    api_key: str
    model: str
    dimensions: int | None = None
    timeout_seconds: float = 45.0

    @classmethod
    def from_environment(cls) -> "EmbeddingSettings | None":
        key = os.getenv("POLICYLENS_EMBEDDING_API_KEY", "").strip()
        if not key:
            return None
        dimensions = os.getenv("POLICYLENS_EMBEDDING_DIMENSIONS")
        return cls(
            base_url=os.getenv("POLICYLENS_EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=key,
            model=os.getenv("POLICYLENS_EMBEDDING_MODEL", "text-embedding-3-small"),
            dimensions=int(dimensions) if dimensions else None,
            timeout_seconds=float(os.getenv("POLICYLENS_EMBEDDING_TIMEOUT_SECONDS", "45")),
        )

