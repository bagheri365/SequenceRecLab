# Milestone 1 — Dataset Semantics and Inclusion Rules

## Purpose

SequenceRecLab asks whether **chronological order itself** improves next-item prediction. That claim is only meaningful when the timestamped events have a defensible behavioral interpretation.

This document freezes the dataset semantics before model implementation. The machine-readable version is [`configs/datasets.toml`](../configs/datasets.toml).

## Inclusion criteria

A dataset may be used for a **primary behavioral claim** only when all of the following are true:

1. Interactions contain timestamps or an equivalent total ordering.
2. The ordering represents actual interaction events rather than a later annotation process.
3. A stable session or user identifier exists within the sequence scope.
4. Item identifiers are available at each interaction.
5. There are enough histories of the required length for the planned comparison.
6. Repeat-item semantics can be stated explicitly for the domain.
7. The preprocessing rule can be reproduced deterministically.

Datasets that fail criterion 2 may still be used as **controlled benchmarks**, but conclusions about real behavioral order must not rely on them alone.

---

## Dataset decisions

### YOOCHOOSE — primary dataset

**Decision:** include as the primary dataset.

The RecSys Challenge 2015 describes YOOCHOOSE as sequences of click events grouped into click sessions, with buying events available for some sessions. The data covers real e-commerce activity. This makes within-session click order directly relevant to the project's central question.

**Sequence scope:** session.

**Primary event:** click.

**Primary target:** next clicked item.

**Repeat-item policy:** allow previously clicked items to remain valid targets. Repeated product inspection can be legitimate within a shopping session.

**Recommended primary history lengths:** `2, 3, 5, 10`.

The previously proposed `5, 20, 50` grid is inappropriate as the main YOOCHOOSE experiment because session histories are typically much shorter than long-term user histories.

**Important limitation:** YOOCHOOSE is session-level. It is a strong dataset for studying short-term order, but it cannot answer questions about long-term user preference trajectories across sessions.

Source: <https://recsys.acm.org/recsys15/challenge/>

---

### Retailrocket — replication dataset

**Decision:** include as the preferred replication dataset.

Retailrocket contains timestamped e-commerce events tied to visitor IDs. Its public data card distinguishes `view`, `addtocart`, and `transaction` events, which gives the dataset stronger behavioral semantics than rating logs and allows histories to extend beyond a single anonymous session.

**Sequence scope:** visitor.

**Primary event for initial experiments:** view.

**Primary target:** next viewed item.

**Repeat-item policy:** allow repeats by default. Revisiting a product is behaviorally meaningful in e-commerce browsing.

**Recommended history lengths:** `5, 20, 50`, subject to an eligibility report showing enough visitors at each threshold.

**Important limitation:** the dataset is sparse across many visitors. Before locking the final primary matrix, preprocessing must report how many visitors survive each minimum-history threshold. If the `50`-interaction cohort is too small or unrepresentative, it must be dropped rather than forced.

Source: <https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>

---

### MovieLens 1M — controlled benchmark only

**Decision:** include as a controlled benchmark, not as the main behavioral dataset.

MovieLens contains timestamped rating events, but rating time is not guaranteed to equal viewing time. A user can rate previously watched movies in a batch. Chronological rating order therefore has weaker behavioral semantics than click or browsing logs.

**Sequence scope:** user.

**Primary event:** rating.

**Primary target:** next rated item.

**Repeat-item policy:** mask previously rated items because MovieLens 1M contains a rating record per user/movie pair rather than a repeated-consumption stream.

**Recommended history lengths:** `5, 20, 50`.

**Allowed claims:** reproducibility, controlled ablations, sensitivity checks.

**Disallowed claim:** MovieLens alone demonstrates that real consumption order causes a sequential-model gain.

Source: <https://grouplens.org/datasets/movielens/1m/>

---

## Primary dataset hierarchy

```text
YOOCHOOSE
  -> primary short-term behavioral-order experiment

Retailrocket
  -> cross-domain / persistent-visitor replication

MovieLens 1M
  -> controlled benchmark and reproducibility check
```

MovieLens 100K may later be added as a development-only fixture, but it is not required for Milestone 1.

---

## Repeat-item policy

The evaluation code must not impose one universal seen-item masking rule.

| Dataset | Default policy | Rationale |
|---|---|---|
| YOOCHOOSE | Allow repeats | Repeated product clicks can be meaningful in-session behavior. |
| Retailrocket | Allow repeats | Revisits are meaningful browsing events. |
| MovieLens 1M | Mask seen items | Ratings are not modeled as repeat-consumption events. |

Any later change to these policies must update both this document and `configs/datasets.toml` in the same commit.

---

## Required profiling before training

The preprocessing milestone must produce, for each dataset:

- number of interactions;
- number of unique sequence identities;
- number of unique items;
- interaction-count distribution per sequence;
- fraction of sequences reaching each planned history length;
- repeat-item rate;
- inter-event-time distribution when timestamps support it;
- dataset time span;
- number of tied timestamps and the deterministic tie-breaking rule.

No model comparison should begin until this profile exists.

---

## Frozen Milestone 1 decisions

Milestone 1 is considered complete when:

- [x] dataset roles are explicit;
- [x] timestamp semantics are explicit;
- [x] identity scope is explicit;
- [x] repeat-item policy is explicit;
- [x] initial history-length grids are explicit;
- [x] weak-order datasets are prevented from carrying the main behavioral claim;
- [x] the decisions are represented in a machine-readable registry;
- [x] tests protect required registry fields and project invariants.
