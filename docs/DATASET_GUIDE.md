# Evaluation dataset guide

The portfolio benchmark should contain 90–100 manually verified questions from
3–5 policy documents that you are legally allowed to redistribute or describe.

## Recommended composition

| Category | Target | Purpose |
|---|---:|---|
| Direct | 20 | Basic evidence retrieval |
| Paraphrased | 25 | Meaning beyond exact wording |
| Multi-passage/page | 15 | Evidence synthesis |
| Table | 15 | Layout and parsing stress |
| Unanswerable | 20 | Abstention and hallucination control |

These are practical targets, not statistical guarantees. Always report the raw
counts and confidence intervals. Avoid strong per-category claims when a
category contains few cases.

## Labelling procedure

1. Read the source and write the expected answer.
2. Record every page that contains necessary evidence.
3. Mark a question answerable only when the evidence is explicit.
4. Ask a second person to review a sample if possible.
5. Keep wording varied; do not simply copy source sentences into questions.
6. Include realistic unanswerable questions that are topically related.
7. Assign the split before experiments begin.

## Leakage rule

- Development cases may be inspected while selecting chunking, retrieval and
  refusal thresholds.
- Test cases are used only after those decisions are frozen.
- Never move a difficult test question into development after seeing results.
- Never report development metrics as the final generalization result.

## Answer grading

Use this rubric:

- `2`: fully correct and complete;
- `1`: directionally correct but incomplete or slightly imprecise;
- `0`: incorrect, contradicted, or unsupported refusal.

Grade faithfulness separately: every material claim must be supported by the
retrieved passages. An LLM judge may perform a first pass, but manually inspect
all failures and a random sample of passes. Record the judging model and prompt.

## Final claim language

Good:

> On 30 held-out questions, hybrid retrieval with reranking placed labelled
> evidence in the top three for 26/30 cases (86.7%; bootstrap interval ...).

Bad:

> Reranking proved the system is 10% better.

