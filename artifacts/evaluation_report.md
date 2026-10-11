# PolicyLens evaluation report

> Generated from the labelled evaluation set. Small differences are suggestive, not proof of universal superiority.

## Retrieval and refusal results

| Configuration | Split | Questions | Hit@1 | Hit@3 | 95% bootstrap CI | MRR | Correct refusal | False refusal | Latency | Cost/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | development | 70 | 81.0% | 90.5% | 82.5%–96.8% | 0.849 | 85.7% | 19.0% | 0.2 ms | $0.000000 |
| hybrid | test | 30 | 66.7% | 77.8% | 63.0%–92.6% | 0.748 | 100.0% | 37.0% | 0.2 ms | $0.000000 |
| hybrid_rerank | development | 70 | 79.4% | 93.7% | 87.3%–98.4% | 0.861 | 85.7% | 19.0% | 0.4 ms | $0.000000 |
| hybrid_rerank | test | 30 | 74.1% | 81.5% | 66.7%–92.6% | 0.772 | 66.7% | 29.6% | 0.4 ms | $0.000000 |
| semantic | development | 70 | 98.4% | 100.0% | 100.0%–100.0% | 0.992 | 100.0% | 14.3% | 0.1 ms | $0.000000 |
| semantic | test | 30 | 88.9% | 100.0% | 100.0%–100.0% | 0.938 | 100.0% | 25.9% | 13.8 ms | $0.000000 |

## Failure analysis

| Configuration | Question type | Failure category | Count |
|---|---|---|---:|
| hybrid | paraphrased | false_refusal | 12 |
| hybrid_rerank | paraphrased | false_refusal | 12 |
| semantic | table | false_refusal | 6 |
| hybrid | table | false_refusal | 5 |
| semantic | paraphrased | false_refusal | 5 |
| hybrid_rerank | table | false_refusal | 4 |
| hybrid | multi_page | false_refusal | 4 |
| hybrid_rerank | table | ranking_failure | 4 |
| hybrid_rerank | multi_page | false_refusal | 3 |
| semantic | multi_page | false_refusal | 3 |
| hybrid_rerank | paraphrased | retrieval_miss | 3 |
| semantic | direct | false_refusal | 2 |
| hybrid | paraphrased | retrieval_miss | 2 |
| hybrid_rerank | unanswerable | unsafe_answer | 2 |
| hybrid_rerank | table | table_extraction_or_retrieval | 2 |
| hybrid | table | table_extraction_or_retrieval | 2 |
| hybrid | table | ranking_failure | 1 |
| hybrid | multi_page | incomplete_multi_passage_retrieval | 1 |
| hybrid | direct | false_refusal | 1 |
| hybrid_rerank | multi_page | ranking_failure | 1 |
| hybrid_rerank | paraphrased | ranking_failure | 1 |
| hybrid | unanswerable | unsafe_answer | 1 |
| hybrid_rerank | direct | false_refusal | 1 |
| semantic | paraphrased | ranking_failure | 1 |

## Interpretation guardrails

- Report raw counts with percentages.
- Select thresholds and configurations on development data only.
- Use held-out results for final claims.
- Manually review answer correctness and a sample of faithfulness grades.
- Do not describe small or overlapping differences as proven improvements.
