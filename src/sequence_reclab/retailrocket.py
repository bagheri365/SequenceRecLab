from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Iterable, Mapping, Sequence

from .profiling import file_sha256
from .yoochoose import build_item_mapping, filter_sessions, temporal_split

RETAILROCKET_EVENTS_SHA256 = "3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7"


@dataclass(frozen=True, slots=True)
class RetailrocketEvent:
    session_id: str
    visitor_id: int
    timestamp: datetime
    item_id: str
    category: str
    row_number: int


@dataclass(frozen=True, slots=True)
class RetailrocketExample:
    session_id: str
    user_id: int
    history: tuple[int, ...]
    target: int
    target_timestamp: str


def read_view_events(path: str | Path) -> dict[int, tuple[RetailrocketEvent, ...]]:
    """Read only Retailrocket view events, grouped by persistent visitor identity."""
    grouped: dict[int, list[RetailrocketEvent]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected Retailrocket schema: {reader.fieldnames!r}")
        for row_number, row in enumerate(reader, start=2):
            if row["event"] != "view":
                continue
            timestamp = datetime.fromtimestamp(int(row["timestamp"]) / 1000, tz=timezone.utc)
            visitor_id = int(row["visitorid"])
            grouped[visitor_id].append(
                RetailrocketEvent(
                    session_id="",
                    visitor_id=visitor_id,
                    timestamp=timestamp,
                    item_id=row["itemid"],
                    category="",
                    row_number=row_number,
                )
            )
    return {
        visitor: tuple(sorted(events, key=lambda event: (event.timestamp, event.row_number)))
        for visitor, events in grouped.items()
    }


def sessionize_views(
    visitors: Mapping[int, Sequence[RetailrocketEvent]],
    *,
    inactivity_minutes: int = 30,
) -> dict[str, tuple[RetailrocketEvent, ...]]:
    """Split each visitor's ordered view stream after gaps strictly greater than the threshold."""
    if inactivity_minutes < 1:
        raise ValueError("inactivity_minutes must be >= 1")
    threshold_seconds = inactivity_minutes * 60
    sessions: dict[str, tuple[RetailrocketEvent, ...]] = {}
    for visitor_id in sorted(visitors):
        events = tuple(sorted(visitors[visitor_id], key=lambda event: (event.timestamp, event.row_number)))
        if not events:
            continue
        chunks: list[list[RetailrocketEvent]] = [[]]
        for event in events:
            if chunks[-1]:
                gap = (event.timestamp - chunks[-1][-1].timestamp).total_seconds()
                if gap > threshold_seconds:
                    chunks.append([])
            chunks[-1].append(event)
        for ordinal, chunk in enumerate(chunks, start=1):
            session_id = f"{visitor_id}:{ordinal}"
            sessions[session_id] = tuple(
                RetailrocketEvent(
                    session_id=session_id,
                    visitor_id=event.visitor_id,
                    timestamp=event.timestamp,
                    item_id=event.item_id,
                    category=event.category,
                    row_number=event.row_number,
                )
                for event in chunk
            )
    return sessions


def generate_examples(
    sessions: Mapping[str, Sequence[RetailrocketEvent]],
    item_to_index: Mapping[str, int],
    *,
    max_history: int | None = None,
    require_fully_known_session: bool = True,
    allowed_user_ids: set[int] | None = None,
) -> list[RetailrocketExample]:
    """Generate next-view examples while retaining persistent visitor IDs for BPR."""
    if max_history is not None and max_history < 1:
        raise ValueError("max_history must be >= 1 or None")
    result: list[RetailrocketExample] = []
    for session_id, seq in sorted(sessions.items()):
        if not seq:
            continue
        visitor_id = seq[0].visitor_id
        if any(event.visitor_id != visitor_id for event in seq):
            raise ValueError(f"session {session_id!r} contains multiple visitor IDs")
        if allowed_user_ids is not None and visitor_id not in allowed_user_ids:
            continue
        if require_fully_known_session and any(event.item_id not in item_to_index for event in seq):
            continue
        encoded = [(item_to_index[event.item_id], event) for event in seq if event.item_id in item_to_index]
        for target_pos in range(1, len(encoded)):
            start = 0 if max_history is None else max(0, target_pos - max_history)
            history = tuple(index for index, _ in encoded[start:target_pos])
            target, target_event = encoded[target_pos]
            result.append(
                RetailrocketExample(
                    session_id=session_id,
                    user_id=visitor_id,
                    history=history,
                    target=target,
                    target_timestamp=target_event.timestamp.isoformat().replace("+00:00", "Z"),
                )
            )
    return result


def write_jsonl(examples: Iterable[RetailrocketExample], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), separators=(",", ":")) + "\n")
            count += 1
    return count


def _summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {"count": len(values), "mean": float(fmean(values)), "median": float(median(values)), "max": float(max(values))}


def profile_retailrocket_sessions(
    sessions: Mapping[str, Sequence[RetailrocketEvent]],
    *,
    history_lengths: Sequence[int] = (2, 3, 5),
) -> dict[str, object]:
    lengths = tuple(sorted(set(history_lengths)))
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("history_lengths must contain positive integers")
    session_lengths: list[int] = []
    items: set[str] = set()
    visitors: set[int] = set()
    repeat_events = 0
    adjacent_repeats = 0
    ties = 0
    gaps: list[float] = []
    for session_id, seq in sessions.items():
        if not seq:
            raise ValueError(f"session {session_id!r} is empty")
        ordered = sorted(seq, key=lambda event: (event.timestamp, event.row_number))
        if list(seq) != ordered:
            raise ValueError(f"session {session_id!r} is not deterministically ordered")
        session_lengths.append(len(seq))
        visitors.add(seq[0].visitor_id)
        items.update(event.item_id for event in seq)
        counts = Counter(event.item_id for event in seq)
        repeat_events += sum(count - 1 for count in counts.values())
        for previous, current in zip(seq, seq[1:]):
            gap = (current.timestamp - previous.timestamp).total_seconds()
            gaps.append(gap)
            adjacent_repeats += previous.item_id == current.item_id
            ties += gap == 0
    interactions = sum(session_lengths)
    transitions = sum(max(0, length - 1) for length in session_lengths)
    return {
        "session_count": len(sessions),
        "visitor_count": len(visitors),
        "interaction_count": interactions,
        "unique_item_count": len(items),
        "session_length": _summary(session_lengths),
        "eligibility": {
            str(length): {
                "sessions_with_at_least_history_plus_target": sum(value >= length + 1 for value in session_lengths),
                "eligible_prefix_examples": sum(max(0, value - length) for value in session_lengths),
            }
            for length in lengths
        },
        "repeat": {
            "repeat_event_count": repeat_events,
            "repeat_event_rate": repeat_events / interactions if interactions else 0.0,
            "adjacent_repeat_transition_count": adjacent_repeats,
            "adjacent_repeat_transition_rate": adjacent_repeats / transitions if transitions else 0.0,
        },
        "timing": {"inter_event_seconds": _summary(gaps), "tied_timestamp_transition_count": ties},
    }


def preprocess_retailrocket(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    inactivity_minutes: int = 30,
    min_session_length: int = 2,
    min_item_support: int = 5,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    max_history: int | None = None,
    expected_sha256: str | None = RETAILROCKET_EVENTS_SHA256,
) -> dict[str, object]:
    """Create deterministic view-only next-item splits with persistent visitor IDs."""
    input_path = Path(input_path)
    actual_sha256 = file_sha256(input_path) if expected_sha256 is not None else None
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"Retailrocket events.csv SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")

    visitors = read_view_events(input_path)
    raw_view_count = sum(len(events) for events in visitors.values())
    sessions = sessionize_views(visitors, inactivity_minutes=inactivity_minutes)
    raw_session_count = len(sessions)
    sessions = filter_sessions(sessions, min_session_length=min_session_length, min_item_support=min_item_support)
    splits = temporal_split(sessions, train_fraction=train_fraction, validation_fraction=validation_fraction)
    item_to_index = build_item_mapping(splits["train"])
    train_user_ids = {event.visitor_id for seq in splits["train"].values() for event in seq}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "item_mapping.json").write_text(json.dumps(item_to_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    primary_example_counts: dict[str, int] = {}
    primary_eligible_sessions: dict[str, int] = {}
    primary_eligible_visitors: dict[str, int] = {}
    returning_example_counts: dict[str, int] = {}
    returning_eligible_sessions: dict[str, int] = {}
    returning_eligible_visitors: dict[str, int] = {}
    for split_name, split_sessions in splits.items():
        # The primary sequential cohort is restricted only by the train item
        # vocabulary. It must not be narrowed merely to accommodate BPR cold start.
        primary_examples = generate_examples(
            split_sessions,
            item_to_index,
            max_history=max_history,
        )
        primary_example_counts[split_name] = write_jsonl(primary_examples, out / f"{split_name}.jsonl")
        primary_eligible_sessions[split_name] = len({example.session_id for example in primary_examples})
        primary_eligible_visitors[split_name] = len({example.user_id for example in primary_examples})

        # BPR can score only persistent visitors observed in training. Emit a
        # separate matched returning-user evaluation cohort so RecentHistoryGain
        # never mixes populations. Training is shared and therefore not duplicated.
        if split_name != "train":
            returning_examples = generate_examples(
                split_sessions,
                item_to_index,
                max_history=max_history,
                allowed_user_ids=train_user_ids,
            )
            returning_example_counts[split_name] = write_jsonl(
                returning_examples, out / f"{split_name}_returning_users.jsonl"
            )
            returning_eligible_sessions[split_name] = len({example.session_id for example in returning_examples})
            returning_eligible_visitors[split_name] = len({example.user_id for example in returning_examples})

    metadata: dict[str, object] = {
        "dataset": "retailrocket",
        "input": {"path": str(input_path), "sha256": actual_sha256, "expected_sha256": expected_sha256},
        "raw_view_count": raw_view_count,
        "raw_view_visitor_count": len(visitors),
        "session_count_before_filtering": raw_session_count,
        "session_count_after_filtering": len(sessions),
        "split_session_counts": {name: len(value) for name, value in splits.items()},
        "primary_cohort": {
            "definition": "train-vocabulary-known sessions; persistent visitor may be unseen in training",
            "eligible_session_counts": primary_eligible_sessions,
            "eligible_visitor_counts": primary_eligible_visitors,
            "example_counts": primary_example_counts,
        },
        "returning_user_cohort": {
            "definition": "held-out primary-cohort sessions whose persistent visitor occurs in training",
            "eligible_session_counts": returning_eligible_sessions,
            "eligible_visitor_counts": returning_eligible_visitors,
            "example_counts": returning_example_counts,
            "files": {
                "validation": "validation_returning_users.jsonl",
                "test": "test_returning_users.jsonl",
            },
        },
        "train_item_count": len(item_to_index),
        "train_persistent_visitor_count": len(train_user_ids),
        "parameters": {
            "primary_event": "view",
            "sessionization": "split_after_inactivity_strictly_greater_than_threshold",
            "inactivity_minutes": inactivity_minutes,
            "min_session_length": min_session_length,
            "min_item_support": min_item_support,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "max_history": max_history,
            "unknown_item_policy": "drop_entire_eval_session",
            "unseen_visitor_policy": "retain_in_primary; restrict_only_separate_returning_user_cohort",
            "id_mapping_scope": "train_only",
            "repeat_items": "retained",
            "seen_item_policy": "allow",
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
