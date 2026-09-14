from __future__ import annotations

import json
from datetime import datetime, timezone

from sequence_reclab.yoochoose import (
    ClickEvent,
    build_item_mapping,
    filter_sessions,
    generate_examples,
    group_sessions,
    parse_timestamp,
    preprocess_yoochoose,
    select_latest_session_fraction,
    temporal_split,
)


def ev(session: str, second: int, item: str, row: int) -> ClickEvent:
    return ClickEvent(
        session_id=session,
        timestamp=datetime(2024, 1, 1, 0, 0, second, tzinfo=timezone.utc),
        item_id=item,
        category="0",
        row_number=row,
    )


def test_timestamp_is_utc_aware():
    ts = parse_timestamp("2014-04-07T10:51:09.277Z")
    assert ts.tzinfo is not None
    assert ts.utcoffset().total_seconds() == 0


def test_group_sessions_restores_chronological_order_with_stable_ties():
    grouped = group_sessions([
        ev("s", 2, "b", 2),
        ev("s", 1, "a", 3),
        ev("s", 1, "c", 1),
    ])
    assert [event.item_id for event in grouped["s"]] == ["c", "a", "b"]


def test_filtering_iterates_until_constraints_hold():
    sessions = {
        "a": (ev("a", 1, "x", 1), ev("a", 2, "y", 2)),
        "b": (ev("b", 3, "x", 3), ev("b", 4, "z", 4)),
    }
    filtered = filter_sessions(sessions, min_session_length=2, min_item_support=2)
    assert filtered == {}


def test_temporal_split_keeps_sessions_whole_and_ordered():
    sessions = {
        f"s{i}": (ev(f"s{i}", i, "x", i), ev(f"s{i}", i + 1, "y", i + 10))
        for i in range(1, 11)
    }
    splits = temporal_split(sessions, train_fraction=0.6, validation_fraction=0.2)
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [6, 2, 2]
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert max(seq[-1].timestamp for seq in splits["train"].values()) <= min(
        seq[-1].timestamp for seq in splits["validation"].values()
    )


def test_item_mapping_is_train_only_and_deterministic():
    train = {
        "s": (ev("s", 1, "10", 1), ev("s", 2, "2", 2), ev("s", 3, "alpha", 3))
    }
    mapping = build_item_mapping(train)
    assert mapping == {"2": 1, "10": 2, "alpha": 3}
    assert 0 not in mapping.values()


def test_examples_preserve_prefix_order_and_drop_unknown_eval_sessions():
    train = {"t": (ev("t", 1, "a", 1), ev("t", 2, "b", 2), ev("t", 3, "c", 3))}
    mapping = build_item_mapping(train)
    valid = {
        "known": (ev("known", 1, "a", 4), ev("known", 2, "b", 5), ev("known", 3, "c", 6)),
        "unknown": (ev("unknown", 1, "a", 7), ev("unknown", 2, "new", 8)),
    }
    examples = generate_examples(valid, mapping, max_history=2)
    assert [x.session_id for x in examples] == ["known", "known"]
    assert examples[-1].history == (mapping["a"], mapping["b"])
    assert examples[-1].target == mapping["c"]


def test_end_to_end_preprocessing_is_deterministic(tmp_path):
    raw = tmp_path / "yoochoose-clicks.dat"
    rows = []
    for session in range(1, 11):
        rows.append(f"{session},2014-04-{session:02d}T10:00:00.000Z,1,0\n")
        rows.append(f"{session},2014-04-{session:02d}T10:01:00.000Z,2,0\n")
        rows.append(f"{session},2014-04-{session:02d}T10:02:00.000Z,3,0\n")
    raw.write_text("".join(rows), encoding="utf-8")

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    metadata1 = preprocess_yoochoose(raw, out1, min_item_support=1, train_fraction=0.6, validation_fraction=0.2)
    metadata2 = preprocess_yoochoose(raw, out2, min_item_support=1, train_fraction=0.6, validation_fraction=0.2)

    assert metadata1 == metadata2 | {"input": str(raw)}
    for name in ["item_mapping.json", "train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json"]:
        assert (out1 / name).read_text() == (out2 / name).read_text()

    meta = json.loads((out1 / "metadata.json").read_text())
    assert meta["parameters"]["id_mapping_scope"] == "train_only"
    assert meta["parameters"]["unknown_item_policy"] == "drop_entire_eval_session"


def test_preprocessing_latest_fraction_matches_explicit_subset_rule(tmp_path):
    raw = tmp_path / "yoochoose-clicks.dat"
    rows = []
    for session in range(1, 9):
        rows.append(f"{session},2014-04-{session:02d}T10:00:00.000Z,1,0\n")
        rows.append(f"{session},2014-04-{session:02d}T10:01:00.000Z,2,0\n")
    raw.write_text("".join(rows), encoding="utf-8")

    metadata = preprocess_yoochoose(
        raw,
        tmp_path / "subset",
        min_item_support=1,
        train_fraction=0.5,
        validation_fraction=0.25,
        latest_session_fraction=0.25,
    )
    assert metadata["raw_session_count"] == 8
    assert metadata["selected_session_count_before_filtering"] == 2
    assert metadata["session_count_after_filtering"] == 2
    assert metadata["parameters"]["latest_session_fraction"] == 0.25
