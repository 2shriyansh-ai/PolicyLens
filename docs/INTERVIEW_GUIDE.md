# PolicyLens interview guide

## Thirty-second explanation

PolicyLens is an evaluated RAG system for company policies. It extracts text
and page metadata, compares two chunking strategies, combines semantic and
keyword retrieval, reranks the strongest candidates, and gives an LLM only the
retrieved evidence. It cites the source or refuses when evidence is weak. I also
built a held-out evaluation harness to measure retrieval, answer quality,
faithfulness, refusal behavior, latency and cost.

## The most important distinction

The retriever finds evidence. The LLM writes one answer from that evidence. The
evaluation harness tests both. Retrieved passages are internal candidates, not
multiple answers shown to the employee.

## Why not just use NotebookLM?

NotebookLM is a strong general-purpose document product. PolicyLens is not a
claim to replace it. The engineering contribution is a transparent,
domain-specific reliability test bench: labelled evidence, configurable
retrieval, explicit abstention, held-out evaluation and reproducible failure
analysis.

## Decisions you must be able to defend

- Why preserve page metadata? For verifiable citations and retrieval labels.
- Why compare chunking? A rule split at the wrong boundary becomes hard to find.
- Why hybrid search? Semantic search handles paraphrases; BM25 handles exact
  identifiers and policy terms.
- Why rerank? Fast retrieval gets candidates; careful scoring improves their
  order.
- Why abstain? A plausible unsupported answer is worse than an honest refusal.
- Why separate development/test? Otherwise threshold and configuration tuning
  makes reported results optimistic.
- Why several metrics? Retrieval, generation, refusal, speed and cost can fail
  independently.

## Honest limitations

- OCR is needed for image-only PDFs.
- Tables remain a likely failure category even with structured extraction.
- Evidence scores are not calibrated probabilities.
- Small benchmark differences are suggestive, not proof.
- A citation can be relevant without fully supporting every generated claim.
- Results apply to the evaluated corpus, not every company policy.

