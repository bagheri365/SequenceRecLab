# Cross-dataset synthesis

Milestone 19 adds a reporting-only layer over frozen YOOCHOOSE and Retailrocket analyses. It does not refit models, alter preprocessing, select checkpoints, or inspect test labels beyond the already generated result artifacts.

`run_cross_dataset_synthesis.py` consumes each dataset's seed-summary CSV and bootstrap CSV and writes four deterministic artifacts under `results/`: `gain_table.csv`, `positional_ndcg10.csv`, `synthesis.md`, and `manuscript_methods_results.md`.

The combined gain table always retains `dataset`, `cohort`, `history_length`, metric, seed count, mean, sample standard deviation, uncertainty method, and interval endpoints. YOOCHOOSE positional intervals are labeled `paired_example_bootstrap`; Retailrocket intervals are labeled `paired_session_cluster_bootstrap`. Missing intervals are explicit rather than silently borrowing an uncertainty method from another gain.

The estimand boundary is enforced in code. YOOCHOOSE cannot contain RecentHistoryGain because session IDs are not persistent users. On Retailrocket, RecentHistoryGain must use the `returning_users` cohort, while ContextGain, PositionalGain, and DeepSeqGain must use the `primary` cohort. The reporting layer never adds gains across those populations.

Example invocation after the frozen per-dataset artifacts exist:

```bash
python scripts/run_cross_dataset_synthesis.py \
  --yoochoose-summary results/yoochoose_1_64/final_analysis/summary.csv \
  --yoochoose-bootstrap results/yoochoose_1_64/final_analysis/positional_bootstrap.csv \
  --retailrocket-summary results/retailrocket/final_analysis/summary.csv \
  --retailrocket-bootstrap results/retailrocket/final_analysis/session_cluster_bootstrap.csv \
  --output-dir results/cross_dataset
```

The exact local YOOCHOOSE paths may differ if earlier frozen artifacts were stored under another results subdirectory; pass those existing files rather than regenerating or retuning models for this reporting milestone.

Milestone 20 extends this reporting layer with the manuscript-ready Methods/Results/Limitations artifact described in `docs/manuscript_package.md`.
