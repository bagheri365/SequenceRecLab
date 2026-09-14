import math

import pytest

from sequence_reclab.evaluation import (
    EvaluationPolicy,
    YOOCHOOSE_POLICY,
    build_candidates,
    evaluate_ranking,
    ndcg_at_k,
    rank_items,
)


def test_yoochoose_keeps_seen_items_and_repeat_target_eligible():
    candidates = build_candidates(
        item_count=5,
        history=(1, 2, 1),
        target=1,
        policy=YOOCHOOSE_POLICY,
    )
    assert candidates == (1, 2, 3, 4, 5)
    assert 0 not in candidates


def test_policy_rejects_masking_when_repeat_targets_are_valid():
    with pytest.raises(ValueError, match="conflicts"):
        EvaluationPolicy(mask_seen_items=True, allow_repeat_targets=True)


def test_target_must_be_in_training_vocabulary():
    with pytest.raises(ValueError, match="training item vocabulary"):
        build_candidates(item_count=5, history=(1, 2), target=6)


def test_ranking_ties_break_by_ascending_item_id():
    ranking = rank_items({1: 0.5, 2: 0.8, 3: 0.8}, (1, 2, 3))
    assert ranking == (2, 3, 1)


def test_single_positive_metrics_have_expected_values():
    ranking = (4, 3, 2, 1)
    assert ndcg_at_k(ranking, target=3, k=10) == pytest.approx(1 / math.log2(3))
    metrics = evaluate_ranking(ranking, target=3, cutoffs=(1, 2))
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 1.0
    assert metrics["ndcg@2"] == pytest.approx(1 / math.log2(3))
    assert metrics["mrr@10"] == 0.5


def test_sampled_candidates_are_deterministic_and_include_target():
    policy = EvaluationPolicy(
        candidate_mode="sampled",
        mask_seen_items=False,
        allow_repeat_targets=True,
        sampled_negative_count=3,
        sampling_seed=17,
    )
    first = build_candidates(
        item_count=10,
        history=(1, 2),
        target=4,
        policy=policy,
        example_key="session-7:3",
    )
    second = build_candidates(
        item_count=10,
        history=(1, 2),
        target=4,
        policy=policy,
        example_key="session-7:3",
    )
    assert first == second
    assert len(first) == 4
    assert 4 in first


def test_candidate_policy_does_not_depend_on_future_items():
    candidates = build_candidates(
        item_count=4,
        history=(1, 2),
        target=3,
        policy=YOOCHOOSE_POLICY,
    )
    assert candidates == (1, 2, 3, 4)
