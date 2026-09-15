import pytest

torch = pytest.importorskip("torch")

from sequence_reclab.evaluation import build_candidates, rank_items
from sequence_reclab.transformers import PositionlessSASRec, SASRec, TransformerConfig


def _config(**overrides):
    values = dict(
        item_count=6,
        max_history_length=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        learning_rate=0.02,
        epochs=3,
        batch_size=2,
        seed=11,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _examples():
    return [
        ((1, 2), 3),
        ((2, 1), 4),
        ((1, 3), 5),
        ((3, 1), 6),
    ]


def test_matched_pair_has_identical_parameter_shapes():
    positionless = PositionlessSASRec(_config())
    positional = SASRec(_config())
    assert positionless.all_parameter_shapes() == positional.all_parameter_shapes()
    assert not positionless.network.position_embedding.weight.requires_grad
    assert positional.network.position_embedding.weight.requires_grad


def test_initial_nonpositional_parameters_are_identical_for_same_seed():
    positionless = PositionlessSASRec(_config())
    positional = SASRec(_config())
    left = positionless.network.state_dict()
    right = positional.network.state_dict()
    for name in left:
        if name == "position_embedding.weight":
            continue
        assert torch.equal(left[name], right[name]), name


def test_positionless_representation_is_permutation_invariant():
    model = PositionlessSASRec(_config())
    first = torch.tensor(model.representation([1, 2, 1]))
    second = torch.tensor(model.representation([2, 1, 1]))
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_positionless_scores_remain_permutation_invariant_after_training():
    model = PositionlessSASRec(_config(epochs=4)).fit(_examples())
    candidates = [1, 2, 3, 4, 5, 6]
    first = model.score([1, 2, 1], candidates)
    second = model.score([2, 1, 1], candidates)
    assert first == pytest.approx(second, abs=1e-6, rel=1e-6)


def test_positional_model_can_distinguish_permutations():
    model = SASRec(_config())
    first = torch.tensor(model.representation([1, 2, 3]))
    second = torch.tensor(model.representation([3, 2, 1]))
    assert not torch.allclose(first, second, atol=1e-7, rtol=1e-7)


def test_both_variants_use_existing_full_catalog_evaluation_contract():
    candidates = build_candidates(item_count=6, target=3, history=[1, 2])
    for cls in (PositionlessSASRec, SASRec):
        model = cls(_config(epochs=1)).fit(_examples())
        scores = model.score([1, 2], candidates)
        ranking = rank_items(scores, candidates)
        assert set(scores) == set(candidates)
        assert set(ranking) == set(candidates)


def test_transformer_rejects_hidden_internal_history_truncation():
    model = PositionlessSASRec(_config(max_history_length=2))
    with pytest.raises(ValueError, match="truncate in the data pipeline"):
        model.score([1, 2, 3], [1, 2, 3])


def test_transformer_rejects_items_outside_training_vocabulary():
    model = SASRec(_config())
    with pytest.raises(ValueError, match="outside item vocabulary"):
        model.score([1, 99], [1, 2])
    with pytest.raises(ValueError, match="outside item vocabulary"):
        model.score([1, 2], [1, 99])


def test_transformer_rejects_nonpositive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        _config(batch_size=0)


def test_fit_never_materializes_more_than_configured_batch_size(monkeypatch):
    model = PositionlessSASRec(_config(batch_size=2, epochs=2))
    observed = []
    original = model._batch

    def recording_batch(examples):
        observed.append(len(examples))
        return original(examples)

    monkeypatch.setattr(model, "_batch", recording_batch)
    model.fit(_examples())
    assert observed
    assert max(observed) <= 2
    assert len(observed) == 4
