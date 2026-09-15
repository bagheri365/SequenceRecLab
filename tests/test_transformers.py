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


def test_transformer_rejects_nonpositive_early_stopping_patience():
    with pytest.raises(ValueError, match="early_stopping_patience"):
        _config(early_stopping_patience=0)


def test_early_stopping_restores_best_validation_checkpoint(monkeypatch):
    model = SASRec(_config(epochs=10, early_stopping_patience=2))
    observed_states = []
    metrics = iter([0.10, 0.30, 0.20, 0.19])

    def scripted_metric(examples):
        observed_states.append(
            {name: value.detach().clone() for name, value in model.network.state_dict().items()}
        )
        return next(metrics)

    monkeypatch.setattr(model, "_validation_ndcg_at_10", scripted_metric)
    model.fit(_examples(), validation_examples=_examples())

    assert model.best_epoch_ == 2
    assert model.epochs_ran_ == 4
    assert model.best_validation_ndcg_at_10_ == pytest.approx(0.30)
    restored = model.network.state_dict()
    for name, value in observed_states[1].items():
        assert torch.equal(restored[name], value), name


def test_validation_ndcg_uses_deterministic_item_id_tie_break():
    model = PositionlessSASRec(_config(item_count=3, epochs=1))
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
    # All items tie. Ascending item id gives target 1 rank 1 and target 3 rank 3.
    value = model._validation_ndcg_at_10([((1,), 1), ((1,), 3)])
    expected = (1.0 + 1.0 / torch.log2(torch.tensor(4.0)).item()) / 2.0
    assert value == pytest.approx(expected)
