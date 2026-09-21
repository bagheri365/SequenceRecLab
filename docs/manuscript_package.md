# Manuscript-ready reporting package

Milestone 20 is reporting-only and remains downstream of the frozen experiments and uncertainty analyses. It adds no model fitting, hyperparameter selection, preprocessing change, or new statistical procedure.

`run_cross_dataset_synthesis.py` now also writes `manuscript_methods_results.md`. The artifact is generated from the same frozen summary and bootstrap CSV inputs as the M19 synthesis, so numerical claims remain traceable to the analysis tables rather than being copied manually into prose.

The generated package contains manuscript-ready Methods, Results, and Limitations sections. It documents the matched positional ablation, validation-only checkpoint selection, three fixed training seeds, the distinct YOOCHOOSE example-level and Retailrocket session-cluster bootstrap units, and the separate Retailrocket returning-user RecentHistoryGain estimand.

The primary cross-dataset claim is intentionally narrow: PositionalGain on NDCG@10 is reported across histories 2, 3, and 5. The text describes the observed predictive pattern without treating the decomposition as universal causal attribution.

Generated manuscript artifacts belong under `results/` and remain untracked.
