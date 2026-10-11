from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import replace
from typing import Protocol

import numpy as np
import requests

from .config import EmbeddingSettings
from .models import Chunk, RetrievedChunk


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-free baseline using hashed word and character features.

    It is intentionally labelled as a baseline, not a production semantic model.
    Install sentence-transformers to use SentenceTransformerEmbedder.
    """

    name = "hashing-baseline"

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                number = int.from_bytes(digest, "little")
                column = number % self.dimensions
                sign = 1.0 if (number >> 1) % 2 else -1.0
                matrix[row, column] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install the 'semantic' extra to use this embedder.") from exc
        self.name = model_name or self.name
        try:
            # Avoid unnecessary network checks after the model has been cached.
            self.model = SentenceTransformer(self.name, local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(self.name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


class OpenAICompatibleEmbedder:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self.name = settings.model

    def encode(self, texts: list[str]) -> np.ndarray:
        body: dict = {"model": self.settings.model, "input": texts}
        if self.settings.dimensions:
            body["dimensions"] = self.settings.dimensions
        response = requests.post(
            f"{self.settings.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        ordered = sorted(response.json()["data"], key=lambda item: item["index"])
        matrix = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = [tokenize(document) for document in documents]
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.term_frequencies = [Counter(document) for document in self.documents]
        document_frequencies: Counter[str] = Counter()
        for document in self.documents:
            document_frequencies.update(set(document))
        count = len(self.documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def scores(self, query: str) -> np.ndarray:
        output = np.zeros(len(self.documents), dtype=np.float32)
        for index, frequencies in enumerate(self.term_frequencies):
            length_normalizer = 1 - self.b + self.b * self.lengths[index] / max(
                self.average_length, 1
            )
            for term in tokenize(query):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                output[index] += self.idf.get(term, 0) * (
                    frequency * (self.k1 + 1)
                    / (frequency + self.k1 * length_normalizer)
                )
        return output


class PolicyIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = chunks
        self.embedder = embedder or HashingEmbedder()
        self.embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        self.keyword = BM25Index([chunk.text for chunk in chunks])
        self._query_embeddings: dict[str, np.ndarray] = {}

    @staticmethod
    def _rank(scores: np.ndarray) -> list[int]:
        return list(np.argsort(-scores, kind="stable"))

    def semantic_search(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if question not in self._query_embeddings:
            self._query_embeddings[question] = self.embedder.encode([question])[0]
        query = self._query_embeddings[question]
        scores = self.embeddings @ query
        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
                semantic_score=float(scores[index]),
            )
            for rank, index in enumerate(self._rank(scores)[:top_k], 1)
        ]

    def keyword_search(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        scores = self.keyword.scores(question)
        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
                keyword_score=float(scores[index]),
            )
            for rank, index in enumerate(self._rank(scores)[:top_k], 1)
        ]

    def hybrid_search(self, question: str, top_k: int = 5, candidate_k: int = 15) -> list[RetrievedChunk]:
        semantic = self.semantic_search(question, candidate_k)
        keyword = self.keyword_search(question, candidate_k)
        semantic_by_id = {item.chunk.id: item for item in semantic}
        keyword_by_id = {item.chunk.id: item for item in keyword}
        fused: dict[str, float] = {}
        # Reciprocal-rank fusion avoids incorrectly adding incomparable raw scores.
        for item in semantic:
            fused[item.chunk.id] = fused.get(item.chunk.id, 0) + 1 / (60 + item.rank)
        for item in keyword:
            fused[item.chunk.id] = fused.get(item.chunk.id, 0) + 1 / (60 + item.rank)
        ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
        chunks_by_id = {chunk.id: chunk for chunk in self.chunks}
        def evidence_confidence(chunk_id: str) -> float:
            semantic_score = (semantic_by_id.get(chunk_id) or RetrievedChunk(chunks_by_id[chunk_id], 0, 0)).semantic_score or 0.0
            keyword_score = (keyword_by_id.get(chunk_id) or RetrievedChunk(chunks_by_id[chunk_id], 0, 0)).keyword_score or 0.0
            # Saturating transform preserves absolute evidence strength. Dividing
            # by the best result would make even an irrelevant query look strong.
            normalized_keyword = keyword_score / (keyword_score + 3.0)
            return 0.55 * max(semantic_score, 0.0) + 0.45 * normalized_keyword

        return [
            RetrievedChunk(
                chunk=chunks_by_id[chunk_id],
                # RRF determines order; this separate value estimates whether the
                # top result contains enough absolute evidence to answer.
                score=evidence_confidence(chunk_id),
                rank=rank,
                semantic_score=(semantic_by_id.get(chunk_id) or RetrievedChunk(chunks_by_id[chunk_id], 0, 0)).semantic_score,
                keyword_score=(keyword_by_id.get(chunk_id) or RetrievedChunk(chunks_by_id[chunk_id], 0, 0)).keyword_score,
            )
            for rank, chunk_id in enumerate(ordered, 1)
        ]

    def rerank(self, question: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Explainable local reranker; production can replace it with a cross-encoder."""
        query_terms = set(tokenize(question))
        scored: list[tuple[float, RetrievedChunk]] = []
        for candidate in candidates:
            terms = set(tokenize(candidate.chunk.text))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            heading_bonus = 0.05 if candidate.chunk.heading and query_terms & set(tokenize(candidate.chunk.heading)) else 0
            score = 0.65 * candidate.score + 0.35 * overlap + heading_bonus
            scored.append((score, candidate))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [replace(candidate, score=float(score), rank=rank) for rank, (score, candidate) in enumerate(scored, 1)]

    def search(self, question: str, method: str = "hybrid_rerank", top_k: int = 5) -> list[RetrievedChunk]:
        if method == "semantic":
            return self.semantic_search(question, top_k)
        if method == "keyword":
            return self.keyword_search(question, top_k)
        if method == "hybrid":
            return self.hybrid_search(question, top_k)
        if method == "hybrid_rerank":
            candidates = self.hybrid_search(question, max(top_k, 10), candidate_k=20)
            return self.rerank(question, candidates)[:top_k]
        raise ValueError(f"Unknown retrieval method: {method}")
