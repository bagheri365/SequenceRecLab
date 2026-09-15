# SequenceRecLab

Controlled experiments on whether chronological interaction order adds predictive value beyond recent-history composition in next-item recommendation.

## Research decomposition

SequenceRecLab separates four effects that are often bundled together as "sequential recommendation":

1. **Recent-history gain** — recent interacted items vs. long-term personalization.
2. **Context gain** — richer interaction modeling without explicit position.
3. **Positional gain** — explicit position in an otherwise matched Transformer.
4. **Deep sequential gain** — long-range sequence modeling vs. a first-order Markov baseline.

## Current milestone

**M9 — real-data protocol lock**

The dataset contract lives in [`configs/datasets.toml`](configs/datasets.toml) and is explained in [`docs/dataset_semantics.md`](docs/dataset_semantics.md). Milestone 2 added deterministic YOOCHOOSE parsing, filtering, temporal splitting, train-only item mapping, and prefix-to-next-item example generation. Milestone 3 froze the repeat-item and ranking-evaluation contract. Milestone 4 added global popularity, first-order Markov transitions, and a deterministic reference BPR matrix-factorization implementation. Milestone 5 added `HistoryPool`, the order-invariant recent-history baseline. Milestone 6 added a matched Transformer pair: `PositionlessSASRec` and position-aware `SASRec`. Both use the same full-prefix self-attention and masked-mean readout; only learned positional information differs, keeping `PositionalGain` interpretable. Milestone 7 adds one experiment runner for seeds, history windows, fixed evaluation cohorts, model execution, metric aggregation, JSONL result records, and the gain decomposition. Milestone 8 adds a real-data audit layer: an explicit latest-session 1/64-style subset rule, eligibility counts, repeat/timing/tied-timestamp diagnostics, exact split manifests, and invariant checks before any model result is trusted. Milestone 9 locks the primary YOOCHOOSE history grid to `[2, 3, 5]` from the observed audit and reserves history 10 for a separately labeled long-session sensitivity analysis.

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
  --min-item-support 5 \
  --latest-session-fraction 0.015625
```

The output contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, `item_mapping.json`, and `metadata.json`. Item IDs are fit on training data only; evaluation sessions containing unseen items are excluded rather than silently deleting unknown events and creating artificial adjacency.


## Evaluation contract

Primary YOOCHOOSE evaluation uses full-catalog ranking over the training vocabulary. Previously clicked items remain eligible because repeated clicks are valid session events. Metrics are Recall@10/20, NDCG@10/20, and MRR@10, with deterministic item-ID tie breaking. Sampled negatives are reserved for explicitly labeled sensitivity analysis. See [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) and [`configs/evaluation.toml`](configs/evaluation.toml).


## Baseline models

Milestones 4–5 add `PopularityModel`, `FirstOrderMarkov`, `BPRMatrixFactorization`, and `HistoryPool`. `HistoryPool` mean-pools training-only item-to-next-target associations over a preselected history window, so permutations of the same history content receive identical scores while repeated items retain multiplicity. The Markov model is directly valid for YOOCHOOSE session sequences. Standard BPR/MF is intentionally guarded against unseen evaluation identities: YOOCHOOSE session IDs are not persistent users, so BPR is reserved for persistent-identity datasets rather than mislabeling session factors as long-term personalization. See [`docs/baselines.md`](docs/baselines.md).


## Matched Transformer pair

Milestone 6 adds `PositionlessSASRec` and `SASRec` in `sequence_reclab.transformers`. The pair shares item embeddings, Transformer dimensions, full-prefix attention, masked-mean readout, tied output embeddings, optimizer, objective, and candidates. A causal mask is intentionally excluded because the mask itself exposes order. See [`docs/transformer_pair.md`](docs/transformer_pair.md) and [`configs/transformers.toml`](configs/transformers.toml).


## Reproducible experiment runner

Milestone 7 centralizes the experimental grid in `sequence_reclab.experiments`. Primary YOOCHOOSE history-length comparisons use `[2, 3, 5]` on one fixed held-out population eligible for history 5, while training retains all valid prefixes and truncates their histories per run. History 10 is a separate long-session sensitivity analysis. Primary YOOCHOOSE runs exclude BPR because sessions are not persistent users. See [`docs/experiment_runner.md`](docs/experiment_runner.md) and [`configs/experiments.toml`](configs/experiments.toml).

Run a primary grid after preprocessing with:

```bash
python scripts/run_experiment.py \
  --processed-dir data/processed/yoochoose \
  --output-dir results/yoochoose_primary
```


## Real-data profiling

Milestone 8 profiles whole YOOCHOOSE sessions before prefix expansion. SequenceRecLab defines its 1/64-style subset as the latest `ceil(N / 64)` sessions by `(session_end_timestamp, session_id)`, followed by iterative support/session filtering and temporal splitting. This is an explicit project preprocessing rule, not a claim that every published `Yoochoose1/64` dataset is identical. See [`docs/yoochoose_profiling.md`](docs/yoochoose_profiling.md).

Run the audit with:

```bash
python scripts/profile_yoochoose.py \
  --input data/raw/yoochoose-clicks.dat \
  --output data/processed/yoochoose_profile \
  --latest-session-fraction 0.015625 \
  --history-lengths 2 3 5 10
```

Inspect `profile.json` before running experiments; `session_manifest.jsonl` records the exact session membership and temporal split.


## Real-data protocol lock

The first real 1/64-style audit retained 100,135 sessions and 432,099 clicks over 8,194 items after filtering. Test eligibility fell from 2,388 sessions at history 5 to 820 at history 10, so history 10 no longer defines the primary comparison population. The exact observed counts and catalog-cohort distinction are recorded in [`docs/real_data_protocol.md`](docs/real_data_protocol.md).
