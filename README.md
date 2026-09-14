# SequenceRecLab

Controlled experiments on whether chronological interaction order adds predictive value beyond recent-history composition in next-item recommendation.

## Research decomposition

SequenceRecLab separates four effects that are often bundled together as "sequential recommendation":

1. **Recent-history gain** — recent interacted items vs. long-term personalization.
2. **Context gain** — richer interaction modeling without explicit position.
3. **Positional gain** — explicit position in an otherwise matched Transformer.
4. **Deep sequential gain** — long-range sequence modeling vs. a first-order Markov baseline.

## Current milestone

**M5 — Order-invariant HistoryPool baseline**

The dataset contract lives in [`configs/datasets.toml`](configs/datasets.toml) and is explained in [`docs/dataset_semantics.md`](docs/dataset_semantics.md). Milestone 2 added deterministic YOOCHOOSE parsing, filtering, temporal splitting, train-only item mapping, and prefix-to-next-item example generation. Milestone 3 froze the repeat-item and ranking-evaluation contract. Milestone 4 added global popularity, first-order Markov transitions, and a deterministic reference BPR matrix-factorization implementation. Milestone 5 adds `HistoryPool`, the order-invariant recent-history baseline used to isolate predictive value from history composition before introducing contextual Transformer models.

The first implementation target is **YOOCHOOSE** because its primary records are real click events grouped into sessions, so event order has direct behavioral meaning. Retailrocket is the preferred replication dataset because it contains timestamped views, add-to-cart events, and transactions tied to persistent visitor IDs. MovieLens 1M is retained only as a controlled benchmark because rating time is not guaranteed to equal consumption time.

## Development

The repository currently requires Python 3.11+ only for the standard-library `tomllib` parser used by the dataset-contract tests.

Run:

```bash
pytest
```

Preprocess a raw `yoochoose-clicks.dat` file with:

```bash
python scripts/preprocess_yoochoose.py \
  --input data/raw/yoochoose-clicks.dat \
  --output data/processed/yoochoose \
  --min-session-length 2 \
  --min-item-support 5
```

The output contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, `item_mapping.json`, and `metadata.json`. Item IDs are fit on training data only; evaluation sessions containing unseen items are excluded rather than silently deleting unknown events and creating artificial adjacency.


## Evaluation contract

Primary YOOCHOOSE evaluation uses full-catalog ranking over the training vocabulary. Previously clicked items remain eligible because repeated clicks are valid session events. Metrics are Recall@10/20, NDCG@10/20, and MRR@10, with deterministic item-ID tie breaking. Sampled negatives are reserved for explicitly labeled sensitivity analysis. See [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) and [`configs/evaluation.toml`](configs/evaluation.toml).


## Baseline models

Milestones 4–5 add `PopularityModel`, `FirstOrderMarkov`, `BPRMatrixFactorization`, and `HistoryPool`. `HistoryPool` mean-pools training-only item-to-next-target associations over a preselected history window, so permutations of the same history content receive identical scores while repeated items retain multiplicity. The Markov model is directly valid for YOOCHOOSE session sequences. Standard BPR/MF is intentionally guarded against unseen evaluation identities: YOOCHOOSE session IDs are not persistent users, so BPR is reserved for persistent-identity datasets rather than mislabeling session factors as long-term personalization. See [`docs/baselines.md`](docs/baselines.md).
