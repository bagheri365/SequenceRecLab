from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    """Frozen next-item evaluation contract for a dataset/domain."""

    candidate_mode: str = "full_catalog"
    mask_seen_items: bool = False
    allow_repeat_targets: bool = True
    padding_index: int = 0
    sampled_negative_count: int | None = None
    sampling_seed: int = 20260914

    def __post_init__(self) -> None:
        if self.candidate_mode not in {"full_catalog", "sampled"}:
            raise ValueError("candidate_mode must be 'full_catalog' or 'sampled'")
        if self.candidate_mode == "sampled":
            if self.sampled_negative_count is None or self.sampled_negative_count < 1:
                raise ValueError("sampled mode requires sampled_negative_count >= 1")
        elif self.sampled_negative_count is not None:
            raise ValueError("sampled_negative_count is only valid in sampled mode")
        if self.mask_seen_items and self.allow_repeat_targets:
            raise ValueError(
                "mask_seen_items=True conflicts with allow_repeat_targets=True: "
                "a valid repeated target could be removed from the candidate set"
            )


YOOCHOOSE_POLICY = EvaluationPolicy(
    candidate_mode="full_catalog",
    mask_seen_items=False,
    allow_repeat_targets=True,
)


def build_candidates(
    *,
    item_count: int,
    target: int,
    history: Sequence[int],
    policy: EvaluationPolicy = YOOCHOOSE_POLICY,
    example_key: str = "",
) -> tuple[int, ...]:
    """Build a deterministic candidate set from the training vocabulary.

    Item indices are assumed to be 1..item_count; index 0 is reserved for padding.
    The held-out target is always required to be part of the training vocabulary.
    """
    if item_count < 1:
        raise ValueError("item_count must be >= 1")
    if not 1 <= target <= item_count:
        raise ValueError("target must belong to the training item vocabulary")

    candidates = set(range(1, item_count + 1))
    if policy.mask_seen_items:
        candidates.difference_update(history)
        candidates.add(target)

    if policy.candidate_mode == "full_catalog":
        return tuple(sorted(candidates))

    negatives = sorted(candidates - {target})
    count = min(policy.sampled_negative_count or 0, len(negatives))
    # Derive a stable per-example seed without Python's process-randomized hash().
    stable_key = sum((index + 1) * ord(char) for index, char in enumerate(example_key))
    rng = random.Random(policy.sampling_seed + stable_key)
    sampled = rng.sample(negatives, count)
    return tuple(sorted([target, *sampled]))


def rank_items(scores: Mapping[int, float], candidates: Iterable[int]) -> tuple[int, ...]:
    """Rank candidates by descending score with deterministic item-ID tie breaking."""
    candidate_list = tuple(candidates)
    missing = [item for item in candidate_list if item not in scores]
    if missing:
        raise ValueError(f"missing scores for candidate items: {missing[:5]}")
    return tuple(sorted(candidate_list, key=lambda item: (-scores[item], item)))


def recall_at_k(ranking: Sequence[int], target: int, k: int) -> float:
    _validate_k(k)
    return float(target in ranking[:k])


def ndcg_at_k(ranking: Sequence[int], target: int, k: int) -> float:
    _validate_k(k)
    try:
        rank = ranking[:k].index(target) + 1
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr_at_k(ranking: Sequence[int], target: int, k: int) -> float:
    _validate_k(k)
    try:
        rank = ranking[:k].index(target) + 1
    except ValueError:
        return 0.0
    return 1.0 / rank


def evaluate_ranking(ranking: Sequence[int], target: int, cutoffs: Sequence[int] = (10, 20)) -> dict[str, float]:
    """Evaluate a single-positive next-item ranking at fixed cutoffs."""
    metrics: dict[str, float] = {}
    for k in cutoffs:
        metrics[f"recall@{k}"] = recall_at_k(ranking, target, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranking, target, k)
    metrics["mrr@10"] = mrr_at_k(ranking, target, 10)
    return metrics


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
