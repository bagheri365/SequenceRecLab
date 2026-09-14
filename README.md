# SequenceRecLab

Controlled experiments on whether chronological interaction order adds predictive value beyond recent-history composition in next-item recommendation.

## Research decomposition

SequenceRecLab separates four effects that are often bundled together as "sequential recommendation":

1. **Recent-history gain** — recent interacted items vs. long-term personalization.
2. **Context gain** — richer interaction modeling without explicit position.
3. **Positional gain** — explicit position in an otherwise matched Transformer.
4. **Deep sequential gain** — long-range sequence modeling vs. a first-order Markov baseline.

## Current milestone

**M1 — Dataset semantics and inclusion rules**

The initial dataset contract lives in [`configs/datasets.toml`](configs/datasets.toml) and is explained in [`docs/dataset_semantics.md`](docs/dataset_semantics.md).

The first implementation target is **YOOCHOOSE** because its primary records are real click events grouped into sessions, so event order has direct behavioral meaning. Retailrocket is the preferred replication dataset because it contains timestamped views, add-to-cart events, and transactions tied to persistent visitor IDs. MovieLens 1M is retained only as a controlled benchmark because rating time is not guaranteed to equal consumption time.

## Development

The repository currently requires Python 3.11+ only for the standard-library `tomllib` parser used by the dataset-contract tests.

Run:

```bash
pytest
```

No model code should be added until the dataset semantics and evaluation policies in Milestone 1 are accepted.
