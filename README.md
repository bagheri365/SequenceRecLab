# SequenceRecLab

Controlled experiments on whether chronological interaction order adds predictive value beyond recent-history composition in next-item recommendation.

## Research decomposition

SequenceRecLab separates four effects that are often bundled together as "sequential recommendation":

1. **Recent-history gain** — recent interacted items vs. long-term personalization.
2. **Context gain** — richer interaction modeling without explicit position.
3. **Positional gain** — explicit position in an otherwise matched Transformer.
4. **Deep sequential gain** — long-range sequence modeling vs. a first-order Markov baseline.

## Current milestone

**M2 — Reproducible YOOCHOOSE sequence pipeline**

The dataset contract lives in [`configs/datasets.toml`](configs/datasets.toml) and is explained in [`docs/dataset_semantics.md`](docs/dataset_semantics.md). Milestone 2 adds deterministic YOOCHOOSE parsing, filtering, temporal splitting, train-only item mapping, and prefix-to-next-item example generation.

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
