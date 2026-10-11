# PolicyLens

PolicyLens is an evaluated retrieval-augmented generation (RAG) system for
company-policy documents. It finds supporting passages, produces one cited
answer, and refuses when its evidence is too weak. A separate Evaluation Lab
compares retrieval configurations and records failures; normal employees only
use the question-answering view.

> The search system finds the evidence. The answerer explains the evidence.
> The evaluation system checks whether both did their jobs.

## Current verified status

The local semantic pipeline, PDF extraction, employee UI and evaluation lab are
working. The controlled benchmark contains five original policy PDFs and 100
labelled questions. The held-out split contains 27 answerable and 3
unanswerable questions:

| Configuration | Answerable cases | Hit@3 | Correct refusals | False refusals |
|---|---:|---:|---:|---:|
| Semantic | 27 | 27/27 | 3/3 | 7/27 |
| Hybrid | 27 | 21/27 | 3/3 | 10/27 |
| Hybrid + local reranking | 27 | 22/27 | 2/3 | 8/27 |

These are controlled-corpus results, not claims about all corporate policies.
They also show why configurations must be measured: semantic retrieval was
strongest, while hybrid retrieval and the local reranker did not improve this
dataset. See the complete [benchmark report](docs/BENCHMARK.md) and the
[resume/interview wording](docs/RESUME.md).

## What is implemented

- PDF and text extraction with document/page metadata.
- Fixed-size and heading/paragraph-aware chunking.
- BM25 keyword retrieval.
- An offline hashing retrieval baseline.
- Optional Sentence Transformer embeddings for real semantic retrieval.
- Optional local FLAN-T5 answer generation with deterministic citations.
- Hybrid retrieval using reciprocal-rank fusion.
- A transparent local reranker.
- Evidence-threshold refusal and source display.
- Development and held-out evaluation splits.
- Hit@1, Hit@3, Hit@5, MRR, correct/false refusal, faithfulness audit,
  latency, raw counts and bootstrap intervals.
- Per-query result logging for failure analysis.
- Streamlit employee view and separate Evaluation Lab.

The repository includes a 100-question controlled benchmark. For stronger
real-world claims, replace or supplement it with independently sourced policy
documents and human-reviewed labels.

## Architecture

```text
PDF/TXT
  -> extract text and page metadata
  -> fixed-size or structure-aware chunks
  -> embedding index + BM25 index

Question
  -> semantic/keyword/hybrid retrieval
  -> optional reranking
  -> evidence threshold
  -> answer or abstain
  -> cited source passages

Labelled questions
  -> run identical configurations
  -> retrieval/refusal/latency metrics
  -> per-query logs and failure analysis
```

## Run locally

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m streamlit run app.py
```

The included text policies work immediately. PDF upload requires PyMuPDF,
which is installed by the standard project dependency command.

For real semantic retrieval:

```powershell
python -m pip install -e ".[semantic]"
python -m streamlit run app.py
```

The first Sentence Transformer use downloads its model. Select **Sentence
Transformer** in the sidebar; the UI labels the offline hashing baseline so it
cannot be mistaken for production semantic search.

## Run the evaluation harness

```powershell
python -m policylens.evaluation `
  --documents data/demo_policies/leave_policy.txt data/demo_policies/travel_policy.txt `
  --cases data/evaluation_cases.csv `
  --chunking structure `
  --embedder sentence-transformer `
  --threshold auto `
  --output artifacts/evaluation_results.csv
```

This creates:

- `evaluation_results.csv`: every question, retrieved chunk, score, decision
  and latency;
- `evaluation_summary.csv`: aggregate metrics with counts and a bootstrap
  confidence interval for Hit@3.
- `evaluation_report.md`: reproducible results and failure-analysis report.
- `manual_grading_template.csv`: the explicit 0/1/2 answer-quality review.
- `failure_summary.csv`: failures grouped by configuration, type and cause.

## Optional model providers

The core project works without credentials. To use an OpenAI-compatible
embedding or chat endpoint, copy `.env.example` into your own environment and
set the `POLICYLENS_*` variables. The app never asks users to paste secrets into
the browser and `.env` is ignored by Git.

```powershell
$env:POLICYLENS_LLM_API_KEY="..."
$env:POLICYLENS_LLM_MODEL="your-model"
$env:POLICYLENS_EMBEDDING_API_KEY="..."
$env:POLICYLENS_EMBEDDING_MODEL="your-embedding-model"
python -m streamlit run app.py
```

The app also offers a cached local `google/flan-t5-small` generator after the
semantic dependencies are installed. It is labelled experimental because a
small local model can be less reliable than the extractive default.

## How the experiments remain fair

1. Use development questions to select chunking, retrieval and threshold.
2. Freeze those settings.
3. Run the held-out test questions without further tuning.
4. Report counts with percentages and confidence intervals.
5. Describe small differences as suggestive, not proven.

`--threshold auto` calibrates a separate threshold for each retrieval method on
the development split. This is necessary because semantic, hybrid and reranked
scores do not share one naturally comparable scale.

## Answer grading

The current offline answerer is extractive: it returns a sentence from the
retrieved evidence, which makes citation faithfulness directly auditable. When
an external generative LLM is added, answer correctness must use the documented
0/1/2 rubric, with an LLM judge only as a first pass and a manual spot-check.
PolicyLens intentionally does not fabricate an automatic “accuracy” number
whose grading method is unclear.

## Current boundaries

- The included local reranker is explainable but is not a neural cross-encoder.
- Scanned PDFs require OCR, which is not yet included.
- Table extraction depends on the PDF text layer and should be treated as a
  dedicated failure category.
- The demo corpus is too small for strong statistical claims.
- No external LLM is required for the MVP; a production answer-generator
  adapter can be added after retrieval quality is measured.

See [PHASES.md](PHASES.md) for the build plan and evidence work.
Use [the dataset guide](docs/DATASET_GUIDE.md) for the final corpus and
[the interview guide](docs/INTERVIEW_GUIDE.md) to learn the system honestly.
