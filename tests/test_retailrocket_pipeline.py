from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from sequence_reclab.retailrocket import (
    RetailrocketEvent,
    generate_examples,
    preprocess_retailrocket,
    read_view_events,
    sessionize_views,
)
from sequence_reclab.yoochoose import build_item_mapping


def event(visitor: int, minute: int, item: str, row: int) -> RetailrocketEvent:
    return RetailrocketEvent(
        session_id="",
        visitor_id=visitor,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute),
        item_id=item,
        category="",
        row_number=row,
    )


def test_reader_keeps_only_views_and_uses_raw_row_as_tie_break(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "timestamp,visitorid,event,itemid,transactionid\n"
        "1000,7,view,20,\n"
        "1000,7,addtocart,99,\n"
        "1000,7,view,10,\n",
        encoding="utf-8",
    )
    views = read_view_events(path)
    assert [value.item_id for value in views[7]] == ["20", "10"]


def test_sessionization_splits_only_when_gap_is_strictly_greater_than_30_minutes():
    visitors = {7: (event(7, 0, "a", 1), event(7, 30, "b", 2), event(7, 61, "c", 3))}
    sessions = sessionize_views(visitors, inactivity_minutes=30)
    assert list(sessions) == ["7:1", "7:2"]
    assert [len(value) for value in sessions.values()] == [2, 1]
    assert all(value.visitor_id == 7 for seq in sessions.values() for value in seq)


def test_examples_preserve_persistent_user_id_and_repeated_targets():
    visitors = {7: (event(7, 0, "a", 1), event(7, 1, "a", 2), event(7, 2, "b", 3))}
    sessions = sessionize_views(visitors)
    mapping = build_item_mapping(sessions)
    examples = generate_examples(sessions, mapping)
    assert [example.user_id for example in examples] == [7, 7]
    assert examples[0].target == mapping["a"]
    assert examples[0].history == (mapping["a"],)


def test_eval_examples_can_require_training_seen_persistent_users():
    sessions = sessionize_views({8: (event(8, 0, "a", 1), event(8, 1, "b", 2))})
    assert generate_examples(sessions, {"a": 1, "b": 2}, allowed_user_ids={7}) == []


def test_preprocessing_writes_user_id_and_is_deterministic_without_hash_check(tmp_path):
    raw = tmp_path / "events.csv"
    rows = ["timestamp,visitorid,event,itemid,transactionid\n"]
    base = 1_700_000_000_000
    # Each visitor contributes two temporally separated sessions; later sessions can
    # therefore remain BPR-compatible after a whole-session temporal split.
    for visitor in range(1, 6):
        for offset in (visitor * 1000, 4_000_000 + visitor * 1000):
            rows.append(f"{base + offset},${visitor},view,1,\n".replace("$", ""))
            rows.append(f"{base + offset + 1000},${visitor},view,2,\n".replace("$", ""))
    raw.write_text("".join(rows), encoding="utf-8")

    out1, out2 = tmp_path / "one", tmp_path / "two"
    first = preprocess_retailrocket(raw, out1, min_item_support=1, train_fraction=0.6, validation_fraction=0.2, expected_sha256=None)
    second = preprocess_retailrocket(raw, out2, min_item_support=1, train_fraction=0.6, validation_fraction=0.2, expected_sha256=None)
    assert first == second | {"input": first["input"]}
    for name in (
        "item_mapping.json",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "validation_returning_users.jsonl",
        "test_returning_users.jsonl",
        "metadata.json",
    ):
        assert (out1 / name).read_text() == (out2 / name).read_text()
    train_rows = [json.loads(line) for line in (out1 / "train.jsonl").read_text().splitlines()]
    assert train_rows and all("user_id" in row and "session_id" in row for row in train_rows)
    assert first["parameters"]["unseen_visitor_policy"] == "retain_in_primary; restrict_only_separate_returning_user_cohort"
    assert first["primary_cohort"]["example_counts"]["test"] >= first["returning_user_cohort"]["example_counts"]["test"]


def test_preprocessing_rejects_wrong_raw_hash(tmp_path):
    raw = tmp_path / "events.csv"
    raw.write_text("timestamp,visitorid,event,itemid,transactionid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preprocess_retailrocket(raw, tmp_path / "out", expected_sha256="0" * 64)


def test_preprocessing_keeps_unseen_visitors_in_primary_but_not_returning_user_cohort(tmp_path):
    raw = tmp_path / "events.csv"
    base = 1_700_000_000_000
    rows = ["timestamp,visitorid,event,itemid,transactionid\n"]
    # Visitor 1 appears in train and later; visitor 2 appears only in the final session.
    sessions = [
        (1, 0),
        (1, 4_000_000),
        (1, 8_000_000),
        (1, 12_000_000),
        (2, 16_000_000),
    ]
    for visitor, offset in sessions:
        rows.append(f"{base + offset},{visitor},view,1,\n")
        rows.append(f"{base + offset + 1000},{visitor},view,2,\n")
    raw.write_text("".join(rows), encoding="utf-8")

    out = tmp_path / "out"
    metadata = preprocess_retailrocket(
        raw, out, min_item_support=1, train_fraction=0.6, validation_fraction=0.2, expected_sha256=None
    )
    primary_test = [json.loads(line) for line in (out / "test.jsonl").read_text().splitlines()]
    returning_test = [json.loads(line) for line in (out / "test_returning_users.jsonl").read_text().splitlines()]
    assert {row["user_id"] for row in primary_test} == {2}
    assert returning_test == []
    assert metadata["primary_cohort"]["example_counts"]["test"] == 1
    assert metadata["returning_user_cohort"]["example_counts"]["test"] == 0
