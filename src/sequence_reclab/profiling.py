from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Mapping, Sequence

from .yoochoose import ClickEvent, select_latest_session_fraction



def _percentile(values: Sequence[float | int], q: float) -> float | None:
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": float(fmean(values)),
        "median": float(median(values)),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)),
    }


def profile_sessions(
    sessions: Mapping[str, Sequence[ClickEvent]],
    *,
    history_lengths: Sequence[int] = (2, 3, 5, 10),
) -> dict[str, object]:
    lengths = tuple(sorted(set(history_lengths)))
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("history_lengths must contain positive integers")

    interaction_count = 0
    items: set[str] = set()
    session_lengths: list[int] = []
    gaps_seconds: list[float] = []
    spans_seconds: list[float] = []
    repeat_events = 0
    adjacent_repeat_transitions = 0
    tied_timestamp_transitions = 0
    sessions_with_ties = 0
    category_switches = 0
    category_transitions = 0

    for session_id, seq in sessions.items():
        if not seq:
            raise ValueError(f"session {session_id!r} is empty")
        ordered = sorted(seq, key=lambda event: (event.timestamp, event.row_number))
        if list(seq) != ordered:
            raise ValueError(f"session {session_id!r} is not deterministically ordered")

        interaction_count += len(seq)
        session_lengths.append(len(seq))
        items.update(event.item_id for event in seq)
        counts = Counter(event.item_id for event in seq)
        repeat_events += sum(count - 1 for count in counts.values())

        tied = False
        for previous, current in zip(seq, seq[1:]):
            gap = (current.timestamp - previous.timestamp).total_seconds()
            if gap < 0:
                raise ValueError(f"session {session_id!r} has decreasing timestamps")
            gaps_seconds.append(gap)
            if gap == 0:
                tied_timestamp_transitions += 1
                tied = True
            if previous.item_id == current.item_id:
                adjacent_repeat_transitions += 1
            if previous.category and current.category:
                category_transitions += 1
                if previous.category != current.category:
                    category_switches += 1
        if tied:
            sessions_with_ties += 1
        spans_seconds.append((seq[-1].timestamp - seq[0].timestamp).total_seconds())

    total_transitions = sum(max(0, length - 1) for length in session_lengths)
    eligibility = {
        str(length): {
            "sessions_with_at_least_history_plus_target": sum(value >= length + 1 for value in session_lengths),
            "eligible_prefix_examples": sum(max(0, value - length) for value in session_lengths),
        }
        for length in lengths
    }

    return {
        "session_count": len(sessions),
        "interaction_count": interaction_count,
        "unique_item_count": len(items),
        "session_length": _summary(session_lengths),
        "eligibility": eligibility,
        "repeat": {
            "repeat_event_count": repeat_events,
            "repeat_event_rate": (repeat_events / interaction_count) if interaction_count else 0.0,
            "adjacent_repeat_transition_count": adjacent_repeat_transitions,
            "adjacent_repeat_transition_rate": (
                adjacent_repeat_transitions / total_transitions if total_transitions else 0.0
            ),
        },
        "timing": {
            "inter_event_seconds": _summary(gaps_seconds),
            "session_span_seconds": _summary(spans_seconds),
            "tied_timestamp_transition_count": tied_timestamp_transitions,
            "sessions_with_tied_timestamps": sessions_with_ties,
        },
        "category": {
            "valid_transition_count": category_transitions,
            "switch_count": category_switches,
            "switch_rate": category_switches / category_transitions if category_transitions else 0.0,
        },
    }


def assert_session_contract(
    sessions: Mapping[str, Sequence[ClickEvent]],
    *,
    min_session_length: int,
    min_item_support: int,
) -> None:
    support = Counter(event.item_id for seq in sessions.values() for event in seq)
    for session_id, seq in sessions.items():
        if len(seq) < min_session_length:
            raise ValueError(f"session {session_id!r} violates min_session_length")
        if list(seq) != sorted(seq, key=lambda event: (event.timestamp, event.row_number)):
            raise ValueError(f"session {session_id!r} is not ordered by timestamp,row_number")
        for event in seq:
            if support[event.item_id] < min_item_support:
                raise ValueError(f"item {event.item_id!r} violates min_item_support")


def assert_split_contract(splits: Mapping[str, Mapping[str, Sequence[ClickEvent]]]) -> None:
    expected = {"train", "validation", "test"}
    if set(splits) != expected:
        raise ValueError(f"expected splits {sorted(expected)}, got {sorted(splits)}")
    identities = {name: set(value) for name, value in splits.items()}
    if identities["train"] & identities["validation"] or identities["train"] & identities["test"] or identities["validation"] & identities["test"]:
        raise ValueError("a session appears in more than one split")

    boundaries: dict[str, tuple[object, object] | None] = {}
    for name, split in splits.items():
        if not split:
            boundaries[name] = None
            continue
        end_times = [seq[-1].timestamp for seq in split.values()]
        boundaries[name] = (min(end_times), max(end_times))
    if boundaries["train"] and boundaries["validation"]:
        if boundaries["train"][1] > boundaries["validation"][0]:
            raise ValueError("train/validation temporal ordering is violated")
    if boundaries["validation"] and boundaries["test"]:
        if boundaries["validation"][1] > boundaries["test"][0]:
            raise ValueError("validation/test temporal ordering is violated")


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_session_manifest(
    splits: Mapping[str, Mapping[str, Sequence[ClickEvent]]],
    path: str | Path,
) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for split_name in ("train", "validation", "test"):
        for session_id, seq in splits[split_name].items():
            rows.append(
                {
                    "session_id": session_id,
                    "split": split_name,
                    "event_count": len(seq),
                    "start_timestamp": seq[0].timestamp.isoformat().replace("+00:00", "Z"),
                    "end_timestamp": seq[-1].timestamp.isoformat().replace("+00:00", "Z"),
                }
            )
    rows.sort(key=lambda row: (row["end_timestamp"], row["session_id"]))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return len(rows)
