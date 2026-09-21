import csv

import pytest

from sequence_reclab.synthesis import (
    PublicationGainRow,
    build_publication_gain_rows,
    validate_estimand_boundaries,
    write_positional_replication_csv,
    write_manuscript_methods_results,
)


def test_build_rows_attaches_only_matching_uncertainty():
    summary = [
        {"cohort": "primary", "kind": "gain", "name": "positional_gain", "history_length": "2", "metric": "ndcg@10", "n_seeds": "3", "mean": "0.02", "sample_sd": "0.001"},
        {"cohort": "primary", "kind": "gain", "name": "context_gain", "history_length": "2", "metric": "ndcg@10", "n_seeds": "3", "mean": "0.01", "sample_sd": "0.002"},
    ]
    bootstrap = [{"history_length": "2", "metric": "ndcg@10", "lower_95": "0.01", "upper_95": "0.03"}]
    rows = build_publication_gain_rows("yoochoose", summary, uncertainty_rows=bootstrap, uncertainty_method="paired_example_bootstrap", uncertainty_gain="positional_gain")
    by_gain = {row.gain: row for row in rows}
    assert by_gain["positional_gain"].lower_95 == pytest.approx(0.01)
    assert by_gain["positional_gain"].uncertainty_method == "paired_example_bootstrap"
    assert by_gain["context_gain"].lower_95 is None
    assert by_gain["context_gain"].uncertainty_method == "none"


def test_retailrocket_estimand_boundary_rejects_cross_cohort_recent_history():
    row = PublicationGainRow("retailrocket", "primary", "recent_history_gain", 2, "ndcg@10", 0.1, 0.0, 3, "none", None, None)
    with pytest.raises(ValueError, match="returning_users"):
        validate_estimand_boundaries([row])


def test_yoochoose_recent_history_is_rejected():
    row = PublicationGainRow("yoochoose", "primary", "recent_history_gain", 2, "ndcg@10", 0.1, 0.0, 3, "none", None, None)
    with pytest.raises(ValueError, match="unavailable"):
        validate_estimand_boundaries([row])


def test_positional_replication_requires_both_datasets_and_preserves_methods(tmp_path):
    rows = []
    for dataset, method in [("yoochoose", "paired_example_bootstrap"), ("retailrocket", "paired_session_cluster_bootstrap")]:
        for history in (2, 3, 5):
            rows.append(PublicationGainRow(dataset, "primary", "positional_gain", history, "ndcg@10", history / 100, 0.001, 3, method, 0.0, 0.1))
    output = tmp_path / "positional.csv"
    write_positional_replication_csv(rows, output)
    with output.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 6
    assert {row["uncertainty_method"] for row in written} == {"paired_example_bootstrap", "paired_session_cluster_bootstrap"}


def test_manuscript_package_preserves_uncertainty_and_cohort_boundaries(tmp_path):
    rows = []
    for dataset, method, means in [
        ("yoochoose", "paired_example_bootstrap", (0.02, 0.04, 0.07)),
        ("retailrocket", "paired_session_cluster_bootstrap", (0.006, 0.011, 0.026)),
    ]:
        for history, mean in zip((2, 3, 5), means):
            rows.append(PublicationGainRow(dataset, "primary", "positional_gain", history, "ndcg@10", mean, 0.002, 3, method, mean - 0.001, mean + 0.001))
    for history in (2, 3, 5):
        rows.append(PublicationGainRow("retailrocket", "returning_users", "recent_history_gain", history, "ndcg@10", 0.17, 0.001, 3, "paired_session_cluster_bootstrap", 0.12, 0.22))
    output = tmp_path / "manuscript.md"
    write_manuscript_methods_results(rows, output)
    text = output.read_text()
    assert "paired example bootstrap" in text
    assert "paired session-cluster bootstrap" in text
    assert "returning-user cohort" in text
    assert "not an additive component" in text
    assert "not universal mechanistic or causal claims" in text


def test_manuscript_package_requires_intervals(tmp_path):
    rows = []
    for dataset in ("yoochoose", "retailrocket"):
        for history in (2, 3, 5):
            rows.append(PublicationGainRow(dataset, "primary", "positional_gain", history, "ndcg@10", 0.01, 0.001, 3, "none", None, None))
    for history in (2, 3, 5):
        rows.append(PublicationGainRow("retailrocket", "returning_users", "recent_history_gain", history, "ndcg@10", 0.17, 0.001, 3, "none", None, None))
    with pytest.raises(ValueError, match="requires frozen 95% intervals"):
        write_manuscript_methods_results(rows, tmp_path / "manuscript.md")
