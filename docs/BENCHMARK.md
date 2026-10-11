# Controlled benchmark results

## Scope

This benchmark uses five original, redistributable, multi-page policy PDFs and
100 labelled questions created for PolicyLens. It is a controlled engineering
benchmark, not a claim about every company's documents.

- 70 development questions
- 30 held-out test questions
- 90 answerable and 10 unanswerable questions
- direct, paraphrased, table, multi-page and unanswerable categories
- local `all-MiniLM-L6-v2` embeddings
- no paid API calls

The configuration and refusal thresholds were selected using development data.
The test results were inspected only after those choices were frozen. After
seeing the held-out results, the settings were not retuned.

## Retrieval comparison

The held-out set contains 27 answerable and 3 unanswerable questions.

| Configuration | Hit@1 | Hit@3 | MRR | Correct refusals | False refusals |
|---|---:|---:|---:|---:|---:|
| Semantic | 24/27 (88.9%) | 27/27 (100%) | 0.938 | 3/3 | 7/27 |
| Hybrid | 18/27 (66.7%) | 21/27 (77.8%) | 0.748 | 3/3 | 10/27 |
| Hybrid + local reranking | 20/27 (74.1%) | 22/27 (81.5%) | 0.772 | 2/3 | 8/27 |

The result does not support a claim that hybrid retrieval or the lightweight
local reranker improved this corpus. Semantic retrieval was strongest. The
refusal threshold generalized poorly: it blocked all unsupported questions for
semantic retrieval, but also refused 7 answerable questions. That is the main
remaining model-quality limitation.

## Chunking ablation

Structure-aware chunking was selected from development data because its MRR was
slightly higher (`0.992` versus `0.984`). On the held-out set, fixed-size chunks
scored slightly higher at Hit@1 and MRR:

| Chunking | Development Hit@1 | Development MRR | Test Hit@1 | Test MRR |
|---|---:|---:|---:|---:|
| Fixed | 61/63 (96.8%) | 0.984 | 25/27 (92.6%) | 0.957 |
| Structure-aware | 62/63 (98.4%) | 0.992 | 24/27 (88.9%) | 0.938 |

The differences are small and the confidence intervals overlap. They are
suggestive, not proof that either strategy is universally better.

## Answer generation

- The default local answerer is extractive and therefore auditable.
- The optional `google/flan-t5-small` generator runs locally and appends a
  deterministic citation, but is labelled experimental.
- OpenAI-compatible chat and embedding adapters are available through
  environment variables.
- The repository emits a manual 0/1/2 grading sheet. No unreviewed automatic
  correctness score is marketed as ground truth.

## Reproduce

```powershell
python scripts/build_benchmark.py
python scripts/run_chunking_ablation.py
python -m policylens.evaluation `
  --documents data/benchmark_policies/benefits_policy.pdf `
              data/benchmark_policies/information_security_policy.pdf `
              data/benchmark_policies/leave_policy.pdf `
              data/benchmark_policies/remote_work_policy.pdf `
              data/benchmark_policies/travel_policy.pdf `
  --cases data/benchmark_cases.csv `
  --chunking structure `
  --embedder sentence-transformer `
  --threshold auto `
  --output artifacts/benchmark_results.csv
```

