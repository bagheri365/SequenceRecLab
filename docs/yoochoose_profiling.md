# YOOCHOOSE profiling contract

Milestone 8 profiles the real event data before any model result is trusted. The profiler operates on whole sessions and never derives statistics from prefix-expanded training examples.

## SequenceRecLab 1/64-style subset

`Yoochoose1/64` is not treated as a unique canonical raw file. SequenceRecLab defines its own reproducible subset:

1. parse and deterministically order each raw session by `(timestamp, raw_row_number)`;
2. sort whole sessions by `(session_end_timestamp, session_id)`;
3. retain the latest `ceil(N / 64)` sessions;
4. inside that selected subset, iteratively remove items below the configured support threshold and sessions below the minimum length until stable;
5. split the surviving sessions temporally into train/validation/test.

This rule is called **SequenceRecLab latest-session 1/64-style**. Results must use that name rather than claiming exact equivalence with every published Yoochoose1/64 preprocessing recipe.

## Required diagnostics

`profile.json` records, for the filtered subset and for every split:

- sessions, interactions, and unique items;
- session-length mean, median, p90, p95, and maximum;
- eligibility for every declared history length, where history length `h` requires at least `h + 1` events for a next-item target;
- repeat-event and adjacent-repeat rates;
- inter-event-time and session-span summaries;
- tied-timestamp counts;
- category-switch rate over adjacent events with category values.

The profiler also records the exact selection/filtering parameters and, unless explicitly disabled, a SHA-256 digest of the raw click file.

`session_manifest.jsonl` is the exact experiment-session manifest. Each row records session ID, split, event count, and start/end timestamps. Commit small derived reports if desired, but do not commit the raw dataset.

## Invariants

Profiling fails instead of silently continuing when:

- a filtered session is shorter than the configured minimum;
- a retained item is below minimum support;
- a session is not ordered by `(timestamp, raw_row_number)`;
- one session appears in multiple splits;
- train/validation/test session-end times violate temporal ordering.

## Recommended first run

```bash
python scripts/profile_yoochoose.py \
  --input data/raw/yoochoose-clicks.dat \
  --output data/processed/yoochoose_profile \
  --latest-session-fraction 0.015625 \
  --min-session-length 2 \
  --min-item-support 5 \
  --history-lengths 2 3 5 10
```

The resulting eligibility counts should be inspected before running the M7 experiment grid. In particular, the fixed-history comparison at 10 requires held-out examples with at least 10 prior clicks, so session-length statistics can make that cohort much smaller than the overall subset.
