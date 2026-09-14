# Baseline model contract

Milestone 4 adds the first model layer without changing the frozen evaluation protocol.

## Popularity

`PopularityModel` scores each candidate by its interaction count in the training data. It is a non-personalized sanity baseline and a fallback signal for unseen Markov source states.

## First-order Markov

`FirstOrderMarkov` estimates

\[
P(i_t=j \mid i_{t-1}=i)
= \frac{C(i \rightarrow j)}{\sum_k C(i \rightarrow k)}.
\]

Only adjacent transitions from training sequences are counted. The model scores from the final item in the observed history. Repeat transitions such as `A -> A` are preserved because repeated clicks are meaningful under the YOOCHOOSE dataset contract. If the final history item has no outgoing transition in training, the model falls back to training-set item popularity.

The tiny popularity term used when a transition row exists only breaks zero-score ties among items with no observed transition. It must remain small enough that any observed transition outranks an unobserved transition.

## BPR matrix factorization

`BPRMatrixFactorization` is a deterministic reference implementation of pairwise Bayesian Personalized Ranking over persistent user-item interactions. It learns user and item latent factors from training-only positive interactions and sampled unobserved training items.

### Identity constraint

A standard user-factor BPR model requires the evaluation user identity to have been observed during training. **YOOCHOOSE does not provide persistent user identities; it provides session IDs.** A temporally held-out YOOCHOOSE session is therefore an unseen identity. Treating each session ID as a long-term user and evaluating its learned factor would either be impossible or leak held-out-session interactions.

For that reason:

- Markov and popularity are valid YOOCHOOSE baselines now.
- BPR/MF is implemented now so the model contract is fixed and testable.
- BPR/MF becomes an empirical personalization baseline on a persistent-identity dataset such as Retailrocket or MovieLens.
- SequenceRecLab must not silently substitute YOOCHOOSE session IDs for persistent users.

This distinction is part of the study design rather than an implementation inconvenience: long-term personalization and within-session order are different sources of predictive information.

## Reproducibility

BPR negative sampling and initialization use a local seeded random generator. Markov and popularity fitting are deterministic. Model scoring returns plain item-to-score mappings and uses the candidate/ranking logic in `sequence_reclab.evaluation`, so candidate policy and tie breaking remain centralized.

## HistoryPool

`HistoryPool` is the deliberately order-invariant recent-history baseline. It is fit from training-only `(history, target)` examples. For each history item `h`, the model estimates the empirical next-target distribution

\[
P(j \mid h) = \frac{C(h, j)}{\sum_k C(h, k)}.
\]

For a supplied history window \(H=(h_1,\ldots,h_m)\), candidate scores are the mean of those per-item association distributions:

\[
S(j \mid H) = \frac{1}{m}\sum_{r=1}^{m} P(j \mid h_r).
\]

This construction has three intentional properties:

- **Permutation invariance:** reordering the same supplied history tokens cannot change the score.
- **Multiplicity preservation:** repeated interactions remain repeated tokens, so composition includes frequency rather than collapsing the history to a set.
- **No hidden recency operation:** `HistoryPool` does not choose the most recent suffix internally. The data/example pipeline selects the history window first; pooling then ignores order inside that fixed window.

An extremely small training-target-popularity term may break zero-score ties without overriding observed history-target associations. With an empty evaluation history the model falls back to training-target popularity, though normal next-item examples in SequenceRecLab have non-empty histories.

### Role in the decomposition

`HistoryPool` is the control for **recent-history composition**. It is not intended to match Transformer capacity. Once the matched positionless Transformer is added, the main comparisons become:

- `RecentHistoryGain = Metric(HistoryPool) - Metric(BPR)` on datasets with persistent user identities.
- `ContextGain = Metric(PositionlessSASRec) - Metric(HistoryPool)`.
- `PositionalGain = Metric(SASRec) - Metric(PositionlessSASRec)`.

For YOOCHOOSE, where BPR cannot represent long-term persistent-user personalization, `HistoryPool` is still directly evaluable and supplies the crucial order-invariant comparator for the later Transformer experiments.
