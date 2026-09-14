import pytest

from sequence_reclab.evaluation import build_candidates, rank_items
from sequence_reclab.models import BPRMatrixFactorization, FirstOrderMarkov, PopularityModel


def test_popularity_counts_training_sequences_only():
    model = PopularityModel().fit([[1, 2, 2], [2, 3]])
    assert model.score([1, 2, 3]) == {1: 1.0, 2: 3.0, 3: 1.0}


def test_markov_uses_last_history_item_and_empirical_transition_probability():
    model = FirstOrderMarkov(fallback_weight=0.0).fit([[1, 2, 3], [4, 2, 3], [2, 1]])
    scores = model.score([9, 2], [1, 3, 4])
    assert scores[3] == pytest.approx(2 / 3)
    assert scores[1] == pytest.approx(1 / 3)
    assert scores[4] == 0.0


def test_markov_preserves_repeat_transitions():
    model = FirstOrderMarkov(fallback_weight=0.0).fit([[1, 1, 2], [1, 1]])
    assert model.transition_count(1, 1) == 2
    assert model.transition_count(1, 2) == 1


def test_markov_falls_back_to_training_popularity_for_unseen_source():
    model = FirstOrderMarkov().fit([[1, 2, 2], [3, 2]])
    ranking = rank_items(model.score([999], [1, 2, 3]), [1, 2, 3])
    assert ranking[0] == 2


def test_markov_scores_full_catalog_candidates_from_evaluation_contract():
    model = FirstOrderMarkov(fallback_weight=0.0).fit([[1, 2, 3], [1, 2, 4]])
    candidates = build_candidates(item_count=4, target=3, history=[1, 2])
    scores = model.score([1, 2], candidates)
    ranking = rank_items(scores, candidates)
    assert set(scores) == set(candidates)
    assert ranking[:2] == (3, 4)  # exact transition tie -> deterministic item-ID tie break


def _bpr_data():
    return [
        (1, 1), (1, 1), (1, 1), (1, 2),
        (2, 3), (2, 3), (2, 3), (2, 4),
        (3, 1), (3, 3),
    ]


def test_bpr_is_deterministic_for_fixed_seed():
    kwargs = dict(factors=4, epochs=15, seed=7, learning_rate=0.03)
    first = BPRMatrixFactorization(**kwargs).fit(_bpr_data())
    second = BPRMatrixFactorization(**kwargs).fit(_bpr_data())
    assert first.score(1, [1, 2, 3, 4]) == second.score(1, [1, 2, 3, 4])


def test_bpr_rejects_unseen_test_identity_instead_of_treating_session_as_user():
    model = BPRMatrixFactorization(factors=2, epochs=2, seed=3).fit(_bpr_data())
    with pytest.raises(KeyError, match="persistent identity"):
        model.score(999, [1, 2, 3, 4])


def test_bpr_rejects_unknown_candidate_item():
    model = BPRMatrixFactorization(factors=2, epochs=2, seed=3).fit(_bpr_data())
    with pytest.raises(KeyError, match="training vocabulary"):
        model.score(1, [1, 99])


def test_bpr_requires_negative_sampling_support():
    with pytest.raises(ValueError, match="unobserved training item"):
        BPRMatrixFactorization(factors=2, epochs=1).fit([(1, 1), (1, 2), (2, 1), (2, 2)])


def _history_pool_data():
    return [
        ((1, 2), 3),
        ((1, 1), 3),
        ((2, 4), 5),
        ((4,), 5),
    ]


def test_history_pool_is_permutation_invariant_for_fixed_history_content():
    from sequence_reclab.models import HistoryPool

    model = HistoryPool(fallback_weight=0.0).fit(_history_pool_data())
    candidates = [1, 2, 3, 4, 5]
    assert model.score([1, 2, 1], candidates) == model.score([2, 1, 1], candidates)


def test_history_pool_preserves_repeat_item_multiplicity():
    from sequence_reclab.models import HistoryPool

    model = HistoryPool(fallback_weight=0.0).fit([((1,), 3), ((2,), 4)])
    first_heavy = model.score([1, 1, 2], [3, 4])
    second_heavy = model.score([1, 2, 2], [3, 4])
    assert first_heavy[3] > second_heavy[3]
    assert first_heavy[4] < second_heavy[4]


def test_history_pool_uses_composition_not_only_final_history_item():
    from sequence_reclab.models import HistoryPool

    model = HistoryPool(fallback_weight=0.0).fit([((1,), 3), ((2,), 4)])
    scores = model.score([1, 2], [3, 4])
    assert scores[3] == pytest.approx(0.5)
    assert scores[4] == pytest.approx(0.5)


def test_history_pool_scores_full_catalog_candidates_from_evaluation_contract():
    from sequence_reclab.models import HistoryPool

    model = HistoryPool(fallback_weight=0.0).fit(_history_pool_data())
    candidates = build_candidates(item_count=5, target=3, history=[1, 2])
    scores = model.score([1, 2], candidates)
    ranking = rank_items(scores, candidates)
    assert set(scores) == set(candidates)
    assert ranking[0] == 3


def test_history_pool_rejects_empty_training_history():
    from sequence_reclab.models import HistoryPool

    with pytest.raises(ValueError, match="non-empty history"):
        HistoryPool().fit([((), 3)])
