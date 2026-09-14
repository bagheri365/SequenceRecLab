from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from sequence_reclab.profiling import (
    assert_session_contract,
    assert_split_contract,
    profile_sessions,
    select_latest_session_fraction,
    write_session_manifest,
)
from sequence_reclab.yoochoose import ClickEvent


def event(session: str, second: int, item: str, row: int, category: str = "A") -> ClickEvent:
    return ClickEvent(
        session_id=session,
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=second),
        item_id=item,
        category=category,
        row_number=row,
    )


def test_latest_fraction_keeps_latest_whole_sessions_and_uses_ceil() -> None:
    sessions = {
        f"s{i}": (event(f"s{i}", i, "1", i), event(f"s{i}", i + 1, "2", i + 10))
        for i in range(5)
    }
    selected = select_latest_session_fraction(sessions, 0.21)
    assert list(selected) == ["s3", "s4"]


def test_latest_fraction_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        select_latest_session_fraction({}, 0)
    with pytest.raises(ValueError):
        select_latest_session_fraction({}, 1.01)


def test_profile_counts_history_eligibility_repeats_ties_and_switches() -> None:
    sessions = {
        "a": (
            event("a", 0, "1", 1, "A"),
            event("a", 0, "1", 2, "A"),
            event("a", 2, "2", 3, "B"),
            event("a", 5, "3", 4, "B"),
        ),
        "b": (
            event("b", 10, "2", 5, "A"),
            event("b", 12, "4", 6, "B"),
            event("b", 13, "2", 7, "B"),
        ),
    }
    profile = profile_sessions(sessions, history_lengths=[2, 3])
    assert profile["session_count"] == 2
    assert profile["interaction_count"] == 7
    assert profile["unique_item_count"] == 4
    assert profile["eligibility"]["2"]["sessions_with_at_least_history_plus_target"] == 2
    assert profile["eligibility"]["2"]["eligible_prefix_examples"] == 3
    assert profile["eligibility"]["3"]["sessions_with_at_least_history_plus_target"] == 1
    assert profile["repeat"]["repeat_event_count"] == 2
    assert profile["repeat"]["adjacent_repeat_transition_count"] == 1
    assert profile["timing"]["tied_timestamp_transition_count"] == 1
    assert profile["timing"]["sessions_with_tied_timestamps"] == 1
    assert profile["category"]["switch_count"] == 2


def test_profile_rejects_unsorted_session() -> None:
    sessions = {"a": (event("a", 2, "1", 2), event("a", 1, "2", 1))}
    with pytest.raises(ValueError, match="not deterministically ordered"):
        profile_sessions(sessions)


def test_session_contract_checks_support() -> None:
    sessions = {
        "a": (event("a", 0, "1", 1), event("a", 1, "2", 2)),
        "b": (event("b", 2, "1", 3), event("b", 3, "3", 4)),
    }
    with pytest.raises(ValueError, match="min_item_support"):
        assert_session_contract(sessions, min_session_length=2, min_item_support=2)


def test_split_contract_rejects_overlap_and_temporal_reversal() -> None:
    a = {"a": (event("a", 0, "1", 1), event("a", 1, "2", 2))}
    b = {"b": (event("b", 2, "1", 3), event("b", 3, "2", 4))}
    assert_split_contract({"train": a, "validation": b, "test": {}})
    with pytest.raises(ValueError, match="more than one split"):
        assert_split_contract({"train": a, "validation": a, "test": {}})
    with pytest.raises(ValueError, match="temporal ordering"):
        assert_split_contract({"train": b, "validation": a, "test": {}})


def test_manifest_is_deterministic_and_records_split(tmp_path) -> None:
    splits = {
        "train": {"b": (event("b", 2, "1", 3), event("b", 3, "2", 4))},
        "validation": {"a": (event("a", 0, "1", 1), event("a", 1, "2", 2))},
        "test": {},
    }
    path = tmp_path / "manifest.jsonl"
    assert write_session_manifest(splits, path) == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["session_id"] for row in rows] == ["a", "b"]
    assert {row["split"] for row in rows} == {"train", "validation"}
