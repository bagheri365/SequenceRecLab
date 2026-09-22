# SequenceRecLab

What does chronological interaction order add beyond the content of recent history in next-item recommendation?

SequenceRecLab is a controlled research project for separating predictive effects that are often bundled together as "sequential recommendation."

The project rule is simple:

> Change one source of sequence information at a time, evaluate matched models on the same held-out population, and keep cohort and uncertainty boundaries explicit.

## At a Glance

- **Research question:** how much predictive value comes from recent-history content, explicit position, and deeper sequence modeling?
- **Datasets:** YOOCHOOSE session clicks and Retailrocket sessionized view events.
- **Primary history lengths:** `2`, `3`, and `5` prior interactions.
- **Primary positional contrast:** `SASRec - PositionlessSASRec`.
- **Main replicated pattern:** PositionalGain on NDCG@10 is positive at all three history lengths on both datasets and grows as more history is available.
- **Magnitude:** YOOCHOOSE `+0.0233 -> +0.0431 -> +0.0720`; Retailrocket `+0.0057 -> +0.0115 -> +0.0262`.
- **Important qualification:** YOOCHOOSE uses an example-level paired bootstrap; Retailrocket uses a session-cluster paired bootstrap. The intervals are not interchangeable, and all bootstrap analyses condition on three fixed training seeds.
- **Code freeze:** `5929de2` (`Add manuscript-ready synthesis reporting`).

## Why This Project Exists

Sequential recommenders can improve for several different reasons at once: they may exploit the set of recent items, explicit positions inside that history, long-range interactions, or persistent-user information. If these mechanisms change together, a performance gain does not reveal which source of information helped.

SequenceRecLab separates four contrasts:

```text
RecentHistoryGain = HistoryPool - BPR
ContextGain       = PositionlessSASRec - HistoryPool
PositionalGain    = SASRec - PositionlessSASRec
DeepSeqGain       = SASRec - Markov
```

The main cross-dataset question is narrower:

> Does explicit positional information add predictive value beyond the unordered content of the same recent interaction history?

## Key Findings

### Positional information replicates across datasets

Final-test NDCG@10 PositionalGain:

| Dataset | h2 | h3 | h5 | Uncertainty |
|---|---:|---:|---:|---|
| YOOCHOOSE | +0.0233 | +0.0431 | +0.0720 | paired example bootstrap |
| Retailrocket | +0.0057 | +0.0115 | +0.0262 | paired session-cluster bootstrap |

All six NDCG@10 intervals exclude zero. The effect is larger on YOOCHOOSE, but the same history-dependent pattern appears on Retailrocket under a different dataset and sessionization protocol.

### The matched positional contrast is not just "Transformer vs. baseline"

`PositionlessSASRec` and `SASRec` share item embeddings, dimensionality, attention heads, depth, feed-forward width, training objective, examples, candidate set, tied output parameterization, full-prefix self-attention, and masked-mean readout.

The positionless control keeps the same-shaped positional table fixed at zero. SASRec learns that table. A causal mask is intentionally not used because the mask itself would expose order to the control.

The resulting PositionalGain is therefore a matched predictive ablation of explicit positional information inside this architecture, not a universal causal estimate of "the value of order."

### Retailrocket shows a different balance of gains

Primary-cohort NDCG@10 mean gains:

| Gain | h2 | h3 | h5 |
|---|---:|---:|---:|
| ContextGain | +0.0423 | +0.0307 | +0.0207 |
| PositionalGain | +0.0057 | +0.0115 | +0.0262 |
| DeepSeqGain | +0.0981 | +0.0915 | +0.0939 |

On Retailrocket, ContextGain decreases as history grows while PositionalGain increases. DeepSeqGain remains large across the three histories. These are predictive contrasts under the frozen protocol; they should not be read as additive causal mechanisms.

### RecentHistoryGain requires a different Retailrocket cohort

BPR requires persistent users. Retailrocket has persistent `visitorid`, so `HistoryPool - BPR` is estimated only on a returning-user cohort:

| History | RecentHistoryGain NDCG@10 | 95% session-cluster interval |
|---:|---:|---:|
| 2 | +0.1785 | [+0.1406, +0.2275] |
| 3 | +0.1791 | [+0.1383, +0.2309] |
| 5 | +0.1678 | [+0.1252, +0.2222] |

This estimand is kept separate from the primary sequential cohort and is not added to the primary decomposition. YOOCHOOSE has no RecentHistoryGain because its session IDs are not persistent user identities.

### Not every metric moves identically

The replicated headline is specifically the NDCG@10 positional pattern. On Retailrocket at history 2, PositionalGain Recall@10 has a 95% session-cluster interval that crosses zero, and Recall@20 has a slightly negative point estimate with an interval that also crosses zero. The evidence should therefore not be summarized as "position improves every metric at every history length."

## Benchmark and Protocol

### YOOCHOOSE

SequenceRecLab uses an explicit latest-session `1/64`-style subset rule rather than assuming equivalence to every published Yoochoose1/64 preprocessing pipeline.

After filtering, the frozen audit contains:

| Quantity | Value |
|---|---:|
| Sessions | 100,135 |
| Interactions | 432,099 |
| Unique items | 8,194 |
| Mean session length | 4.3152 |
| Repeat-event rate | 21.39% |
| Adjacent-repeat rate | 16.48% |

Primary evaluation uses session clicks, retains repeats, does not mask seen items, fits the item vocabulary on training data only, uses a temporal split, and ranks the full training catalog.

### Retailrocket

The primary task uses `view` events only. Events are grouped into sessions using a boundary when inactivity is strictly greater than 30 minutes; ties are ordered by timestamp and original raw row index. Repeats remain valid targets and seen items are not masked.

After filtering, the frozen profile contains:

| Quantity | Value |
|---|---:|
| Sessions | 303,627 |
| Interactions | 1,074,426 |
| Visitors | 248,439 |
| Unique items | 46,739 |
| Mean session length | 3.5386 |
| Repeat-event rate | 27.57% |
| Adjacent-repeat rate | 24.89% |

The primary cohort allows held-out visitors unseen during training because the sequential models do not require persistent training identities. A separate returning-user cohort is used only for the matched BPR/HistoryPool comparison.

## Statistical Evidence

All reported model means use the three predeclared training seeds `17`, `29`, and `43`.

For YOOCHOOSE PositionalGain, uncertainty is a deterministic paired example-level percentile bootstrap over the fixed test examples. This preserves matched model comparisons but does not preserve dependence among examples originating from the same session.

For Retailrocket, gain uncertainty is a deterministic paired session-cluster percentile bootstrap. Whole derived sessions are resampled and the same sampled clusters are used for each matched model pair.

In both cases, training seeds are fixed rather than resampled. The intervals therefore describe evaluation-population uncertainty conditional on those three trained runs; they are **not** confidence intervals over arbitrary retraining seeds.

## Model Selection

Transformer selection is validation-only:

- selection metric: full-catalog `NDCG@10`
- maximum epochs: `50`
- patience: `5`
- `min_delta = 0`
- best validation checkpoint restored before final test evaluation
- same selection machinery for PositionlessSASRec and SASRec

Retailrocket Transformer validation naturally early-stopped across all seed/history/model fits, with best epochs between 18 and 29. Hyperparameters were frozen before final test evaluation.

## Research Evolution

> semantics -> deterministic data -> matched models -> frozen validation -> final test -> uncertainty -> replication -> synthesis

| Phase | Question | Outcome |
|---|---|---|
| M1-M3 | What exactly is the task and evaluation contract? | Session semantics, preprocessing, repeat handling, and full-catalog ranking frozen |
| M4-M6 | Can the proposed gains be represented by matched baselines? | Markov, BPR, HistoryPool, PositionlessSASRec, and SASRec implemented |
| M7-M9 | Can comparisons share one reproducible runner and real-data cohort? | Fixed histories `[2,3,5]`, seeds, cohorts, and profiling protocol |
| M10-M13 | Are sequence reconstruction and Transformer training trustworthy? | Reconstruction bug fixed; deterministic minibatching, early stopping, and provenance added |
| M14-M15 | Does YOOCHOOSE PositionalGain survive paired uncertainty analysis? | Positive NDCG@10 intervals at h2/h3/h5 under paired example bootstrap |
| M16-M18 | Does the pattern replicate on Retailrocket with persistent-user semantics? | Replication protocol, cohort-aware routing, and session-cluster bootstrap frozen |
| M19-M20 | Can both datasets be synthesized without crossing estimand boundaries? | Cross-dataset tables and manuscript-ready reporting generated from frozen results |

## Reproducibility

Run the test suite:

```bash
pytest
```

Rebuild the frozen YOOCHOOSE summary from final-test result records:

```bash
python scripts/run_statistical_analysis.py \
  --results \
    results/yoochoose_1_64/classical_test_3seed/results.jsonl \
    results/yoochoose_1_64/transformers_test_3seed/results.jsonl \
  --output results/yoochoose_1_64/final_analysis/summary.csv
```

Rebuild the cross-dataset reporting layer:

```bash
python scripts/run_cross_dataset_synthesis.py \
  --yoochoose-summary results/yoochoose_1_64/final_analysis/summary.csv \
  --yoochoose-bootstrap results/yoochoose_1_64/positional_gain_bootstrap.csv \
  --retailrocket-summary results/retailrocket/final_analysis/summary.csv \
  --retailrocket-bootstrap results/retailrocket/final_analysis/session_cluster_bootstrap.csv \
  --output-dir results/cross_dataset
```

The reporting command does not refit models, select checkpoints, change preprocessing, or tune against test results.

## Repository Structure

```text
SequenceRecLab/
├── configs/                 # dataset, evaluation, experiment, and Transformer contracts
├── docs/                    # protocol, methodology, and synthesis documentation
├── scripts/                 # preprocessing, experiments, bootstrap, and reporting CLIs
├── src/sequence_reclab/     # implementation
├── tests/                   # deterministic regression and unit tests
└── README.md                # fast-reader research overview
```

Generated `results/` and raw/processed data are intentionally excluded from version control.

## Experimental Discipline

The project keeps the following boundaries explicit:

- train-only item vocabularies
- temporal train/validation/test splits
- validation-only Transformer checkpoint selection
- final test evaluation only after the selection rule is frozen
- identical held-out examples for matched gain calculations
- full-catalog primary ranking
- repeated items retained as meaningful targets
- no seen-item masking
- primary and returning-user cohorts never mixed
- deterministic bootstrap seeds and recorded provenance
- descriptive seed SDs kept distinct from evaluation-population bootstrap intervals

## Limitations

- Only three training seeds are used.
- YOOCHOOSE example-level bootstrap intervals do not preserve within-session dependence.
- The YOOCHOOSE and Retailrocket uncertainty procedures use different sampling units.
- RecentHistoryGain is available only on Retailrocket returning users.
- The primary histories are short (`2`, `3`, `5`); history 10 is a separately labeled YOOCHOOSE sensitivity setting rather than part of the main comparison.
- The matched Transformer pair is SASRec-style rather than a byte-for-byte reproduction of canonical causal-mask SASRec.
- Results establish predictive contrasts under the frozen architectures and protocols, not universal causal or mechanistic claims about sequence order.

## What the Experiments Suggest

The strongest replicated result is narrow but consistent:

> Explicit positional information adds predictive value beyond the unordered content of the same recent interaction history on both YOOCHOOSE and Retailrocket, and the NDCG@10 increment grows from history 2 to history 5 in both datasets.

The magnitude is dataset-dependent. Retailrocket also shows that richer order-invariant context and deeper sequence modeling can contribute substantial predictive value, while the relative contribution of explicit position grows with the available history.

The useful design lesson is therefore not "always use position" or "sequence models always win." It is:

> Separate recent-history content, explicit position, persistent-user information, and deeper sequence modeling before attributing a recommender's gain to sequence order.

## Future Research

- replicate the matched positional ablation on additional session datasets
- increase the number of training seeds when compute permits
- use cluster-aware uncertainty for YOOCHOOSE if stable session identifiers are carried through the per-example evaluation artifact
- test longer histories on datasets with enough eligible sessions without changing the primary frozen analysis
- examine whether the history-dependent positional pattern changes across repeat-heavy, category-switching, or other behavior-defined session strata
- compare additional order-aware architectures while preserving a matched positionless control

## Current Conclusion

SequenceRecLab asks:

> What does chronological order add beyond knowing which items appeared recently?

The frozen evidence says:

> On NDCG@10, explicit positional information adds predictive value in both studied datasets, and that incremental value becomes larger as the observed history grows from 2 to 5 interactions.

That pattern replicates across YOOCHOOSE and Retailrocket, but its magnitude and metric-level behavior are dataset-dependent. The project therefore treats "sequential gain" not as one number, but as a set of matched predictive contrasts that should be measured separately.
