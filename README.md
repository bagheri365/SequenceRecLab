# SequenceRecLab

Controlled experiments on whether chronological interaction order adds predictive value beyond recent-history composition in next-item recommendation.

## Research decomposition

SequenceRecLab separates four effects that are often bundled together as "sequential recommendation":

1. **Recent-history gain** — recent interacted items vs. long-term personalization.
2. **Context gain** — richer interaction modeling without explicit position.
3. **Positional gain** — explicit position in an otherwise matched Transformer.
4. **Deep sequential gain** — long-range sequence modeling vs. a first-order Markov baseline.

## Current milestone

**M6 — Matched PositionlessSASRec + SASRec pair**

The dataset contract lives in [`configs/datasets.toml`](configs/datasets.toml) and is explained in [`docs/dataset_semantics.md`](docs/dataset_semantics.md). Milestone 2 added deterministic YOOCHOOSE parsing, filtering, temporal splitting, train-only item mapping, and prefix-to-next-item example generation. Milestone 3 froze the repeat-item and ranking-evaluation contract. Milestone 4 added global popularity, first-order Markov transitions, and a deterministic reference BPR matrix-factorization implementation. Milestone 5 added `HistoryPool`, the order-invariant recent-history baseline. Milestone 6 adds a matched Transformer pair: `PositionlessSASRec` and position-aware `SASRec`. Both use the same full-prefix self-attention and masked-mean readout; only learned positional information differs, keeping `PositionalGain` interpretable.

The first implementation target is **YOOCHOOSE** because its primary records are real click events grouped into sessions, so event order has direct behavioral meaning. Retailrocket is the preferred replication dataset because it contains timestamped views, add-to-cart events, and transactions tied to persistent visitor IDs. MovieLens 1M is retained only as a controlled benchmark because rating time is not guaranteed to equal consumption time.

## Development

The core pipeline requires Python 3.11+. Transformer experiments use the optional PyTorch dependency installed through the `deep` extra.

Run core tests:

```bash
pytest
```

Install the Transformer extra before M6 experiments:

```bash
python -m pip install -e ".[dev,deep]"
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


## Matched Transformer pair

Milestone 6 adds `PositionlessSASRec` and `SASRec` in `sequence_reclab.transformers`. The pair shares item embeddings, Transformer dimensions, full-prefix attention, masked-mean readout, tied output embeddings, optimizer, objective, and candidates. A causal mask is intentionally excluded because the mask itself exposes order. See [`docs/transformer_pair.md`](docs/transformer_pair.md) and [`configs/transformers.toml`](configs/transformers.toml).
