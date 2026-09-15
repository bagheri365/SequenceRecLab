import json

import pytest

from sequence_reclab.experiments import (
    ExperimentExample,
    ExperimentResult,
    TransformerRunConfig,
    compute_gains,
    fixed_evaluation_population,
    load_examples_jsonl,
    reconstruct_sequences,
    run_grid,
    truncate_examples,
    write_results_jsonl,
)


def _examples():
    return [
        ExperimentExample("s1:1", (1,), 2),
        ExperimentExample("s1:2", (1, 2), 3),
        ExperimentExample("s1:3", (1, 2, 3), 4),
        ExperimentExample("s2:4", (2,), 3),
        ExperimentExample("s2:5", (2, 3), 4),
        ExperimentExample("s2:6", (2, 3, 4), 1),
    ]


def test_fixed_evaluation_population_is_shared_across_history_lengths():
    fixed = fixed_evaluation_population(_examples(), [1, 2, 3])
    assert [example.example_id for example in fixed] == ["s1:3", "s2:6"]
    short = truncate_examples(fixed, 1)
    long = truncate_examples(fixed, 3)
    assert [example.example_id for example in short] == [example.example_id for example in long]
    assert short[0].history == (3,)
    assert long[0].history == (1, 2, 3)


def test_fixed_population_fails_instead_of_silently_changing_cohort():
    with pytest.raises(ValueError, match="no evaluation examples"):
        fixed_evaluation_population(_examples(), [10])


def test_reconstruct_sequences_counts_each_session_once():
    assert reconstruct_sequences(_examples()) == [(1, 2, 3, 4), (2, 3, 4, 1)]


def test_reconstruct_sequences_recovers_events_before_capped_history():
    examples = [
        ExperimentExample("long:10", (1,), 2),
        ExperimentExample("long:11", (1, 2), 3),
        ExperimentExample("long:12", (2, 3), 4),
        ExperimentExample("long:13", (3, 4), 5),
    ]
    assert reconstruct_sequences(examples) == [(1, 2, 3, 4, 5)]


def test_reconstruct_sequences_uses_jsonl_order_not_input_order():
    examples = [
        ExperimentExample("long:13", (3, 4), 5),
        ExperimentExample("long:10", (1,), 2),
        ExperimentExample("long:12", (2, 3), 4),
        ExperimentExample("long:11", (1, 2), 3),
    ]
    assert reconstruct_sequences(examples) == [(1, 2, 3, 4, 5)]


def test_load_examples_jsonl_preserves_optional_persistent_identity(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps({"user_id": 7, "history": [1, 2], "target": 3}) + "\n",
        encoding="utf-8",
    )
    example = load_examples_jsonl(path)[0]
    assert example.identity_id == 7
    assert example.history == (1, 2)


def test_classical_runner_is_deterministic_and_uses_same_eval_population():
    train = _examples()
    evaluation = [
        ExperimentExample("e1:1", (1, 2, 3), 4),
        ExperimentExample("e2:2", (2, 3, 4), 1),
        ExperimentExample("too-short:3", (1,), 2),
    ]
    kwargs = dict(
        dataset="toy",
        train_examples=train,
        evaluation_examples=evaluation,
        item_count=4,
        history_lengths=[1, 3],
        seeds=[17],
        models=["popularity", "markov", "history_pool"],
    )
    first = run_grid(**kwargs)
    second = run_grid(**kwargs)
    assert first == second
    assert {row.example_count for row in first} == {2}
    assert len(first) == 6


def test_bpr_runner_refuses_session_ids_as_persistent_users():
    with pytest.raises(ValueError, match="persistent user_id"):
        run_grid(
            dataset="toy",
            train_examples=_examples(),
            evaluation_examples=[ExperimentExample("e:1", (1, 2), 3)],
            item_count=4,
            history_lengths=[2],
            seeds=[1],
            models=["bpr"],
        )


def _row(model, value, *, count=10):
    return ExperimentResult(
        dataset="toy",
        split="test",
        model=model,
        seed=1,
        history_length=3,
        example_count=count,
        metrics={"ndcg@10": value, "recall@10": value + 0.1},
    )


def test_gain_decomposition_uses_matched_rows_only():
    gains = compute_gains(
        [
            _row("bpr", 0.10),
            _row("markov", 0.20),
            _row("history_pool", 0.30),
            _row("positionless_sasrec", 0.35),
            _row("sasrec", 0.40),
        ]
    )
    by_name = {row.gain: row for row in gains}
    assert by_name["recent_history_gain"].metrics["ndcg@10"] == pytest.approx(0.20)
    assert by_name["context_gain"].metrics["ndcg@10"] == pytest.approx(0.05)
    assert by_name["positional_gain"].metrics["ndcg@10"] == pytest.approx(0.05)
    assert by_name["deep_seq_gain"].metrics["ndcg@10"] == pytest.approx(0.20)


def test_gain_decomposition_rejects_mismatched_populations():
    with pytest.raises(ValueError, match="mismatched populations"):
        compute_gains([_row("history_pool", 0.3, count=9), _row("positionless_sasrec", 0.4, count=10)])


def test_result_writer_is_stable_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    count = write_results_jsonl([_row("markov", 0.2)], path)
    assert count == 1
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["model"] == "markov"
    assert raw["metrics"]["ndcg@10"] == 0.2


def test_transformer_pair_runs_under_same_runner_when_torch_available():
    pytest.importorskip("torch")
    results = run_grid(
        dataset="toy",
        train_examples=_examples(),
        evaluation_examples=[
            ExperimentExample("e1:1", (1, 2), 3),
            ExperimentExample("e2:2", (2, 3), 4),
        ],
        item_count=4,
        history_lengths=[2],
        seeds=[5],
        models=["positionless_sasrec", "sasrec"],
        transformer=TransformerRunConfig(
            hidden_dim=4,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
            learning_rate=0.01,
            epochs=1,
        ),
    )
    assert [row.model for row in results] == ["positionless_sasrec", "sasrec"]
    assert all(row.example_count == 2 for row in results)
