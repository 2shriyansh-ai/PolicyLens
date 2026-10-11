# PolicyLens build phases

## Phase 1 — Foundation

- Domain models for documents, chunks, retrieved passages and answers.
- Reproducible project configuration and an offline demo path.
- Status: complete.

## Phase 2 — Document preparation

- PDF and text extraction with document/page metadata.
- Fixed-size and structure-aware chunking.
- Status: complete.

## Phase 3 — Retrieval

- Semantic-style local retrieval baseline.
- Local Sentence Transformer semantic retrieval.
- BM25 keyword retrieval.
- Hybrid retrieval using reciprocal-rank fusion.
- Candidate reranking and explainable scores.
- Status: complete; the local model and the production API adapter are both
  available.

## Phase 4 — Answering and abstention

- Evidence threshold and refusal behavior.
- One final answer with document/page citations.
- Extractive offline answerer plus pluggable LLM boundary.
- Status: complete for the MVP.

## Phase 5 — Evaluation

- Development/test split support.
- Hit@1/3/5, MRR, refusal rates, latency and faithfulness checks.
- Per-query logs and CSV output for failure analysis.
- Status: complete; deterministic metrics, manual grading templates and an
  optional LLM-judge boundary are included.

## Phase 6 — Product UI

- Employee-facing Ask PolicyLens page.
- Separate Evaluation Lab for engineers and recruiters.
- Status: complete.

## Phase 7 — Evidence and hardening

- Build five original, redistributable policy PDFs.
- Create 100 labelled questions and freeze the 70/30 split.
- Run final experiments, bootstrap confidence intervals and failure analysis.
- Add local and external LLM boundaries after retrieval is measured.
- Status: complete for the controlled benchmark. Independent real-world corpus
  validation remains a future extension, not an unreported claim.
