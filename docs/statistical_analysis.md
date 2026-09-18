# Frozen multi-seed analysis

After the YOOCHOOSE validation protocol is frozen, `run_statistical_analysis.py` combines model-result JSONL files, derives every gain whose required matched model pair is present, and aggregates rows across seeds using the arithmetic mean and sample standard deviation. It does not refit or select models.

The primary uncertainty analysis is a paired example-level percentile bootstrap for `SASRec - PositionlessSASRec`. For each history length, both models are refit with the frozen seeds and validation-based early stopping, then scored on the same fixed test examples. Each bootstrap replicate resamples test-example indices with replacement and preserves the pairing between models and across the three fixed training seeds. The statistic is the mean paired gain across examples and then across the fixed seeds. Seeds are not resampled; with only three predeclared seeds, the interval is explicitly an example-level uncertainty interval, not a population-of-training-seeds interval.

The bootstrap is deterministic from the declared base bootstrap seed. To give each history length a separate reproducible random stream, the effective seed is `base_bootstrap_seed + history_length`. Bootstrap CSV output records both `base_bootstrap_seed` and `effective_bootstrap_seed`, so the command-line seed and the per-history RNG stream are unambiguous. Test labels are used only for final scoring and bootstrap uncertainty estimation, never checkpoint selection or tuning. Generated CSV files belong under `results/` and remain ignored by Git.

For session-only YOOCHOOSE, `recent_history_gain = HistoryPool - BPR` remains unavailable because BPR requires persistent user identities; the analysis must not substitute session IDs for users. With Markov, HistoryPool, PositionlessSASRec, and SASRec present, the report derives ContextGain, PositionalGain, and DeepSeqGain.

## Retailrocket session-cluster bootstrap

Retailrocket uses a session-cluster paired percentile bootstrap for the frozen test decomposition. The sampling unit is the 30-minute view session, not the individual next-view example: each replicate samples test session IDs with replacement and includes every eligible prediction example from each sampled session. This preserves within-session dependence while retaining exact pairing between the two models in each gain comparison. The same sampled session clusters are used across the three fixed training seeds, and training seeds are not resampled.

The primary cohort estimates `context_gain`, `positional_gain`, and `deep_seq_gain`. The separate returning-user cohort estimates only `recent_history_gain`; its intervals are not combined additively with primary-cohort intervals because the populations differ. The script refits models with the already frozen training configuration and validation-based Transformer checkpoint selection, then uses test labels only for final per-example scoring and bootstrap uncertainty. It does not tune preprocessing, models, history lengths, or early stopping from test results.

As in the earlier YOOCHOOSE analysis, the bootstrap is deterministic. The declared base bootstrap seed is recorded alongside the effective per-history seed `base_bootstrap_seed + history_length`. The CSV also records cohort, gain, number of examples, and number of sampled session clusters. These are session-cluster uncertainty intervals conditional on the fixed training seeds, not population-of-training-seeds confidence intervals.

Run the frozen Retailrocket analysis with:

```bash
python scripts/run_retailrocket_cluster_bootstrap.py \
  --processed-dir data/processed/retailrocket \
  --output results/retailrocket/final_analysis/cluster_bootstrap.csv \
  --history-lengths 2 3 5 \
  --seeds 17 29 43 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260918
```
