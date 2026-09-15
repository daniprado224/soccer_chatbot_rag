# RAG Evaluation Report

Retrieval depth: top-5. 26 questions.

## Retrieval metrics (answerable questions only)

| Category | n | Precision@k | Recall@k |
|---|---|---|---|
| ambiguous | 10 | 0.20 | 0.60 |
| easy | 10 | 0.18 | 0.50 |
| **overall** | 20 | 0.19 | 0.55 |

## Generation metrics (separate from retrieval)

| Category | n | Citation overlap | Citation exact | Answerability correct |
|---|---|---|---|---|
| ambiguous | 10 | 0.50 | 0.40 | 0.40 |
| easy | 10 | 0.20 | 0.20 | 0.20 |
| unanswerable | 6 | n/a | n/a | 1.00 |
| **overall** | 26 | 0.35 | 0.30 | 0.46 |

## Failure attribution

Every non-passing question is bucketed by where in the pipeline it broke, so a low score can be traced to retrieval or generation:

- `retrieval`: the correct law was never in the top-k at all (recall@k = 0)
- `generation`: the correct law WAS retrieved, but the final answer didn't cite it
- `over_refusal`: an answerable question was refused ("I don't know")
- `hallucination`: an out-of-corpus question was answered instead of refused
- `llm_error`: the grading or generation API call itself failed (rate limit, quota, network) -- NOT a real model judgment, and should be re-run rather than trusted

| Failure mode | count |
|---|---|
| over_refusal | 14 |
| none (passing) | 12 |

## Per-question detail

| id | category | expected law(s) | cited law(s) | recall@k | answerable (exp/got) | failure |
|---|---|---|---|---|---|---|
| q01 | easy | 3 | 3 | 1.00 | True/True | none |
| q02 | easy | 1 | - | 1.00 | True/False | over_refusal |
| q03 | easy | 2 | - | 0.00 | True/False | over_refusal |
| q04 | easy | 7 | - | 0.00 | True/False | over_refusal |
| q05 | easy | 14 | - | 1.00 | True/False | over_refusal |
| q06 | easy | 5 | - | 1.00 | True/False | over_refusal |
| q07 | easy | 4 | 4 | 1.00 | True/True | none |
| q08 | easy | 15 | - | 0.00 | True/False | over_refusal |
| q09 | easy | 16 | - | 0.00 | True/False | over_refusal |
| q10 | easy | 17 | - | 0.00 | True/False | over_refusal |
| q11 | ambiguous | 12 | 12 | 1.00 | True/True | none |
| q12 | ambiguous | 11 | 11 | 1.00 | True/False | over_refusal |
| q13 | ambiguous | 11 | - | 0.00 | True/False | over_refusal |
| q14 | ambiguous | 5 | - | 0.00 | True/False | over_refusal |
| q15 | ambiguous | 12 | 12 | 1.00 | True/True | none |
| q16 | ambiguous | 12 | - | 0.00 | True/False | over_refusal |
| q17 | ambiguous | 14 | - | 1.00 | True/False | over_refusal |
| q18 | ambiguous | 9 | - | 0.00 | True/False | over_refusal |
| q19 | ambiguous | 12 | 12,13 | 1.00 | True/True | none |
| q20 | ambiguous | 11 | 11 | 1.00 | True/True | none |
| q21 | unanswerable | - | - | n/a | False/False | none |
| q22 | unanswerable | - | - | n/a | False/False | none |
| q23 | unanswerable | - | - | n/a | False/False | none |
| q24 | unanswerable | - | - | n/a | False/False | none |
| q25 | unanswerable | - | - | n/a | False/False | none |
| q26 | unanswerable | - | - | n/a | False/False | none |
