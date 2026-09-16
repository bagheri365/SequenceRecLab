# Retailrocket replication protocol

M16 freezes the Retailrocket data protocol before any Retailrocket model result is observed.

## Raw contract

The primary raw input is `events.csv` from the Retailrocket Kaggle dataset. Its expected SHA-256 is:

`3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7`

The observed schema is `timestamp,visitorid,event,itemid,transactionid`. The primary pipeline reads only `view` rows. `addtocart` and `transaction` are excluded from the primary target, and category/item-property files are not model inputs.

## Sequence semantics

`visitorid` is a persistent identity, not a session ID. For the sequential task, each visitor's view events are ordered by `(timestamp, raw row number)` and split after inactivity **strictly greater than 30 minutes**. Derived session IDs are deterministic `<visitorid>:<ordinal>` values. The primary task is next-viewed-item prediction within one derived session.

Repeated items remain valid targets and evaluation does not mask seen items. The primary history lengths are 2, 3, and 5, matching the frozen YOOCHOOSE comparison.

## Filtering and temporal split

Sessions and items are iteratively filtered with minimum session length 2 and minimum item support 5. Whole derived sessions are then sorted by session end time and split 80/10/10 into train/validation/test. Item IDs are mapped from training only. Evaluation sessions containing any item outside the training vocabulary are excluded in full rather than deleting unknown events and creating artificial adjacency.

The primary sequential validation/test cohort is restricted by the training item vocabulary only; it is **not** narrowed to returning visitors merely to accommodate BPR. This cohort is used for `ContextGain`, `PositionalGain`, and `DeepSeqGain`, preserving the ordinary later-session estimand and the closest comparison with YOOCHOOSE.

BPR cannot score a genuinely unseen persistent user, and session IDs must never be substituted for users. The preprocessor therefore emits separate `validation_returning_users.jsonl` and `test_returning_users.jsonl` files containing only primary-cohort sessions whose `visitorid` occurs in training. `RecentHistoryGain = HistoryPool - BPR` must evaluate **both** HistoryPool and BPR on this same returning-user cohort. Results from the two cohorts must never be subtracted from one another. Every emitted example carries `user_id=visitorid` in addition to its derived `session_id`.

## Pre-result descriptive evidence

The protocol was chosen before model training. The raw event stream contains 2,664,312 views. A view-only 30-minute probe produced 1,756,275 sessions, including 367,652 sessions with at least two views. Among within-session adjacent view pairs, 23.83% repeat the same item and about 0.067% have tied timestamps. These observations support retaining repeats and deterministic tie handling.

## Commands

Profile the frozen protocol:

```bash
python scripts/profile_retailrocket.py \
  --input data/raw/retailrocket/events.csv \
  --output data/profiles/retailrocket
```

Create experiment-ready files:

```bash
python scripts/preprocess_retailrocket.py \
  --input data/raw/retailrocket/events.csv \
  --output data/processed/retailrocket
```

Generated data and profiles are research outputs and should not be committed.

## Runner cohort contract

Experiment result rows carry an explicit `cohort` field. For Retailrocket, the
runner accepts the sequential models on `primary`; BPR is forbidden there.
`returning_users` is reserved for the matched BPR/HistoryPool comparison used by
`RecentHistoryGain`. The experiment CLI selects `validation.jsonl`/`test.jsonl`
for `--cohort primary` and the corresponding `*_returning_users.jsonl` files for
`--cohort returning_users`.

Gain matching includes cohort identity in addition to dataset, split, seed, and
history length. If both sides of a declared gain are supplied only on different
cohorts, analysis fails rather than silently subtracting or merging them.

Example primary run:

```bash
python scripts/run_experiment.py \
  --processed-dir data/processed/retailrocket \
  --output-dir results/retailrocket/primary_validation \
  --dataset retailrocket \
  --split validation \
  --cohort primary \
  --history-lengths 2 3 5 \
  --seeds 17 29 43 \
  --models markov history_pool positionless_sasrec sasrec
```

Returning-user BPR comparison:

```bash
python scripts/run_experiment.py \
  --processed-dir data/processed/retailrocket \
  --output-dir results/retailrocket/returning_validation \
  --dataset retailrocket \
  --split validation \
  --cohort returning_users \
  --history-lengths 2 3 5 \
  --seeds 17 29 43 \
  --models bpr history_pool
```
