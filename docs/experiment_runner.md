# Experiment runner contract

Milestone 7 centralizes experiment execution so model comparisons cannot silently use different seeds, history populations, candidate sets, or metric aggregation.

## Fixed evaluation population

History-length comparisons use a fixed held-out cohort. For the primary YOOCHOOSE experiment, the declared lengths are `[2, 3, 5]`, so an evaluation example is eligible only when its original prefix contains at least 5 interactions. The same eligible examples are then truncated to the most recent 2, 3, or 5 items. This prevents history length from being confounded with a changing evaluation population.

Training is different: every valid training prefix remains available. Its history is truncated to the requested window. This avoids discarding most short-prefix supervision while keeping the held-out comparison population fixed.

## Classical fitting inputs

The M2 JSONL files contain every prefix-to-next-item example. `PopularityModel` and `FirstOrderMarkov` must not treat each prefix as an independent sequence because that would repeatedly count early session events. The runner reconstructs one longest observed sequence per session for these two baselines.

`HistoryPool` and the Transformer pair train directly from prefix-to-next-item examples because their objective is defined at the example level.

## Persistent identity and BPR

The runner supports BPR only when examples carry an explicit persistent `user_id`. A YOOCHOOSE `session_id` is never substituted for a user identity. The primary YOOCHOOSE matrix therefore excludes BPR; `RecentHistoryGain = HistoryPool - BPR` becomes available on a persistent-identity replication dataset.

## Reproducibility

Every result row records:

- dataset and split,
- model,
- random seed,
- history length,
- evaluated example count,
- aggregate ranking metrics,
- for Transformer rows, checkpoint-selection provenance: best epoch, epochs trained, selection metric, best validation metric, maximum epoch budget, and patience.

Rows are serialized as deterministic JSONL. Gain rows are only computed between model results with matching dataset, split, seed, history length, and evaluated population size.

## Gain decomposition

For every compatible metric:

```text
RecentHistoryGain = HistoryPool - BPR
ContextGain       = PositionlessSASRec - HistoryPool
PositionalGain    = SASRec - PositionlessSASRec
DeepSeqGain       = SASRec - Markov
```

Missing model pairs produce no gain row rather than inventing a substitute baseline.

## Real-data history decision

The M8 audit showed that history 10 selects a much smaller long-session population than history 5. SequenceRecLab therefore treats history 10 as a separately labeled sensitivity analysis rather than using it to define the primary fixed cohort. See `docs/real_data_protocol.md`.

## Transformer checkpoint selection

Transformer runs load `validation.jsonl` for checkpoint selection even when `--split test` is requested. The default rule is best full-catalog NDCG@10 with a maximum of 50 epochs and patience 5; the best validation checkpoint is restored before the requested split is evaluated. `--transformer-epochs` sets the maximum epoch budget and `--transformer-patience` sets patience. Each Transformer result row records the selected epoch, total epochs trained, selection metric, best validation value, maximum epoch budget, and patience so checkpoint selection is auditable. Validation-run metrics are tuning diagnostics; final claims should come from the untouched test split after the selection protocol is locked.
