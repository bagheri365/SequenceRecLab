# Evaluation protocol

This document freezes the evaluation contract **before model implementation**.

## Task

SequenceRecLab evaluates **single-positive next-item prediction**. For every prefix, the next observed event is the only relevant target. The primary metrics are Recall@10, Recall@20, NDCG@10, NDCG@20, and MRR@10.

## YOOCHOOSE repeat-item policy

For YOOCHOOSE, previously clicked items remain eligible candidates. A shopper can click the same item more than once within a session, so a repeated next click is a valid target. Masking all previously seen items would make such targets impossible to recommend and would silently change the task.

Therefore the primary YOOCHOOSE policy is:

```text
allow_repeat_targets = true
mask_seen_items = false
```

This policy is **domain-specific**. It must not be copied automatically to later datasets such as MovieLens or Retailrocket; each dataset requires its own semantic decision.

## Candidate set

Primary results use **full-catalog ranking over the training item vocabulary**:

```text
candidate items = {1, ..., number_of_training_items}
```

Index `0` is reserved for padding and is never a candidate. Validation/test sessions containing an item outside the training vocabulary are already excluded by the preprocessing contract, so every evaluated target must be rankable.

Full-catalog ranking is the primary protocol because sampled negatives can materially change measured ranking quality and even model ordering.

## Sampled-negative sensitivity analysis

Sampled evaluation is secondary only. If CPU/runtime constraints justify a sensitivity run, use 100 negatives sampled uniformly from eligible training-vocabulary items with seed `20260914`. The positive target is always included.

Sampled-negative results must be labeled as sampled and must never replace the full-catalog primary table.

## Metric definitions

For a target at one-indexed rank `r`:

```text
Recall@K = 1 if r <= K else 0
NDCG@K   = 1 / log2(r + 1) if r <= K else 0
MRR@K    = 1 / r if r <= K else 0
```

Dataset-level metrics are arithmetic means over evaluated examples. Because there is one relevant target per example, Recall@K is equivalent to HitRate@K; the repository uses the name Recall@K consistently.

## Deterministic ranking

Models provide scores for all candidates. Candidates are sorted by descending score. Exact score ties are broken by ascending integer item ID. This makes metric computation deterministic across runs and implementations.

## Leakage safeguards

Evaluation code must never:

- build the item vocabulary from validation/test data;
- remove a valid repeated target because it appeared in history;
- include padding index `0` as a candidate;
- construct candidates using future events;
- silently evaluate unknown targets;
- change candidate-set policy between models in the same comparison.

The tests in `tests/test_evaluation.py` enforce these invariants.
