import pytest

from sequence_reclab.statistical_analysis import paired_example_bootstrap, summarize_seed_rows


def test_seed_summary_uses_sample_sd():
    rows = [
        {"split": "test", "model": "sasrec", "seed": 1, "history_length": 2, "metrics": {"ndcg@10": 0.2}},
        {"split": "test", "model": "sasrec", "seed": 2, "history_length": 2, "metrics": {"ndcg@10": 0.4}},
    ]
    summary = summarize_seed_rows(rows)[0]
    assert summary.mean == pytest.approx(0.3)
    assert summary.sample_sd == pytest.approx(2 ** 0.5 / 10)
    assert summary.n_seeds == 2


def test_paired_bootstrap_is_deterministic_and_preserves_positive_gain():
    deltas = {
        17: [{"ndcg@10": 0.1}, {"ndcg@10": 0.2}, {"ndcg@10": 0.3}],
        29: [{"ndcg@10": 0.2}, {"ndcg@10": 0.3}, {"ndcg@10": 0.4}],
        43: [{"ndcg@10": 0.15}, {"ndcg@10": 0.25}, {"ndcg@10": 0.35}],
    }
    first = paired_example_bootstrap(deltas, history_length=2, bootstrap_samples=500, bootstrap_seed=9)
    second = paired_example_bootstrap(deltas, history_length=2, bootstrap_samples=500, bootstrap_seed=9)
    assert first == second
    interval = first[0]
    assert interval.base_bootstrap_seed == 9
    assert interval.effective_bootstrap_seed == 9
    assert interval.observed_mean_gain == pytest.approx(0.25)
    assert 0 < interval.lower_95 <= interval.observed_mean_gain <= interval.upper_95


def test_paired_bootstrap_rejects_mismatched_example_populations():
    with pytest.raises(ValueError, match="same non-empty paired example population"):
        paired_example_bootstrap(
            {1: [{"ndcg@10": 0.1}], 2: [{"ndcg@10": 0.2}, {"ndcg@10": 0.3}]},
            history_length=2,
        )


def test_paired_bootstrap_records_base_and_effective_seed_separately():
    deltas = {17: [{"ndcg@10": 0.1}], 29: [{"ndcg@10": 0.2}]}
    interval = paired_example_bootstrap(
        deltas,
        history_length=5,
        bootstrap_samples=10,
        bootstrap_seed=20260921,
        base_bootstrap_seed=20260916,
    )[0]
    assert interval.base_bootstrap_seed == 20260916
    assert interval.effective_bootstrap_seed == 20260921
