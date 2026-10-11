# PolicyLens

PolicyLens is a question-answering app for company policy documents. Give it a
set of PDFs, ask a question in plain English, and it finds the most relevant
passages before returning one answer with the document name and page number.
If the documents do not contain enough evidence, it says so instead of making
up an answer.

I built this project to explore a question that is easy to overlook in RAG
applications: **how do we know the system actually works?** Alongside the main
employee-facing app, PolicyLens has an Evaluation Lab that compares retrieval
methods, measures their results, and records where they fail.

## What it looks like in practice

An employee might ask:

> How many unused leave days can I carry forward?

PolicyLens searches the uploaded policies, returns the answer, and shows the
supporting passage from the correct page. Behind the scenes, the Evaluation Lab
can run the same set of labelled questions through different configurations so
their performance can be compared fairly.

## Main features

- Upload and search PDF policy documents.
- Keep document names, page numbers, and section information with every passage.
- Choose between fixed-size and structure-aware text chunking.
- Compare BM25 keyword search, semantic search, hybrid search, and local reranking.
- Return cited answers or refuse when the evidence is too weak.
- Build and validate a labelled evaluation dataset from the app.
- Measure Hit@1, Hit@3, Hit@5, MRR, faithfulness, refusal behaviour, latency,
  estimated cost, and bootstrap confidence intervals.
- Save per-question results for straightforward failure analysis.
- Run locally without an API key; external model providers are optional.

## How it works

```text
Policy PDFs
    ↓
Extract text while keeping page information
    ↓
Split the text into searchable passages
    ↓
Convert passages into embeddings and build a search index
    ↓
Retrieve the passages most related to the user's question
    ↓
Answer with a citation — or refuse if the evidence is weak
```

The user sees one final answer, not a list of competing answers. The retrieved
passages are shown underneath as sources so the answer can be checked.

## Benchmark results

The repository includes five original sample policy PDFs and 100 labelled test
questions. Settings were chosen using a development split, then evaluated once
on a held-out split containing 27 answerable and 3 unanswerable questions.

| Retrieval method | Hit@3 | Correct refusals | False refusals |
|---|---:|---:|---:|
| Semantic | 27/27 | 3/3 | 7/27 |
| Hybrid | 21/27 | 3/3 | 10/27 |
| Hybrid + local reranking | 22/27 | 2/3 | 8/27 |

Semantic search performed best on this dataset. It found the correct evidence
within its first three results for every answerable held-out question. However,
it also refused 7 answerable questions, which shows that the refusal threshold
still needs improvement.

These are results from the included controlled benchmark—not a claim that the
same numbers will hold for every company's documents. The full methodology,
metrics, and failure breakdown are in [the benchmark report](docs/BENCHMARK.md).

## Run it locally

```powershell
git clone https://github.com/2shriyansh-ai/PolicyLens.git
cd PolicyLens
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[semantic]"
python -m streamlit run app.py
```

The first use of the Sentence Transformer downloads its model. Once the app
opens, keep **Use included demo policies** enabled and ask a question such as:

> How many unused leave days may be carried forward?

The basic application can also be installed with `python -m pip install -e .`.
That version includes the offline hashing baseline but not the Sentence
Transformer or local FLAN-T5 dependencies.

## Run the evaluation

The Evaluation Lab is available inside the Streamlit app. The command-line
harness can also be run directly:

```powershell
python -m policylens.evaluation `
  --documents data/demo_policies/leave_policy.txt data/demo_policies/travel_policy.txt `
  --cases data/evaluation_cases.csv `
  --chunking structure `
  --embedder sentence-transformer `
  --threshold auto `
  --output artifacts/evaluation_results.csv
```

It produces detailed query results, a metric summary, a reproducible Markdown
report, a manual grading sheet, and a grouped failure report. Automatic
threshold selection uses only the development questions, so the held-out test
questions remain genuinely unseen during tuning.

## Optional model providers

No credentials are needed for the default workflow. To use an OpenAI-compatible
embedding or chat service, set the variables documented in `.env.example`:

```powershell
$env:POLICYLENS_LLM_API_KEY="..."
$env:POLICYLENS_LLM_MODEL="your-model"
$env:POLICYLENS_EMBEDDING_API_KEY="..."
$env:POLICYLENS_EMBEDDING_MODEL="your-embedding-model"
python -m streamlit run app.py
```

Secrets are read from the environment and `.env` files are ignored by Git.
There is also an experimental local `google/flan-t5-small` answer generator,
although the extractive answerer remains the more predictable default.

## Limitations and next steps

- The controlled corpus is intentionally small and should be expanded with
  independently reviewed, real-world documents.
- Scanned PDFs need OCR, which is not included yet.
- Complex PDF tables can lose their structure during extraction.
- The local reranker is transparent and lightweight, not a neural cross-encoder.
- Refusal calibration is the clearest area for improvement.

For more detail, see the [dataset guide](docs/DATASET_GUIDE.md),
[interview guide](docs/INTERVIEW_GUIDE.md), and
[resume wording](docs/RESUME.md).
