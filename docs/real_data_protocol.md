# Real-data protocol lock

Milestone 9 records the first real YOOCHOOSE audit and converts the observed eligibility distribution into an explicit experiment decision. These values are provenance for the current SequenceRecLab preprocessing rule, not universal YOOCHOOSE statistics.

## Audited subset

The raw `yoochoose-clicks.dat` file had SHA-256 `0deaf041a4f60955928fe8430e525a1dbf38e9013c7482420973e4aa492fa762`. The SequenceRecLab latest-session fraction was `0.015625` (1/64-style), followed by minimum session length 2 and minimum item support 5.

After filtering, the audit contained 100,135 sessions, 432,099 interactions, and 8,194 unique items. The temporal split contained 80,108 train sessions, 10,013 validation sessions, and 10,014 test sessions.

## History eligibility

Eligibility means a prefix with `h` prior interactions plus one next-item target. In the complete filtered test split:

| History | Eligible examples | Eligible sessions |
| ---: | ---: | ---: |
| 2 | 27,681 | 6,098 |
| 3 | 21,583 | 4,318 |
| 5 | 14,170 | 2,388 |
| 10 | 6,178 | 820 |

History 10 therefore selects a substantially smaller long-session population. Because the experiment runner fixes the evaluation cohort at the largest declared history, including 10 in the primary grid would force every 2/3/5 comparison onto those 820 sessions.

**Protocol decision:** primary YOOCHOOSE history lengths are `[2, 3, 5]`. History `10` is a separately labeled long-history sensitivity analysis with its own fixed cohort. Results from the two cohorts must not be mixed into one history-length trend.

## Repeat and timing diagnostics

The filtered subset had a repeat-event rate of 0.2139 and adjacent-repeat transition rate of 0.1648, supporting the existing YOOCHOOSE policy that previously clicked items remain valid candidates. Median inter-event time was about 61.1 seconds and median session span about 205.3 seconds. Only 29 of 100,135 sessions contained tied timestamps (33 tied transitions), so deterministic raw-row tie breaking remains sufficient.

## Train-vocabulary evaluation cohort

The preprocessing stage fits item IDs on training data only. It produced 8,183 train-vocabulary items and 258,310 train examples. Validation retained 9,995 of 10,013 split sessions and 35,749 examples; test retained 9,961 of 10,014 split sessions and 36,845 examples. Entire held-out sessions containing unseen items are excluded rather than deleting unknown events and creating artificial adjacency.

Reports must distinguish the **complete temporal split** used by the profiler from the **train-vocabulary experiment cohort** emitted by preprocessing. Neither should be silently substituted for the other.

## First experiment order

Run inexpensive baselines first: Popularity, first-order Markov, and HistoryPool on the primary `[2, 3, 5]` grid. Verify result counts and cohort invariants before spending CPU on PositionlessSASRec and SASRec.
