from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True, slots=True)
class ClickEvent:
    session_id: str
    timestamp: datetime
    item_id: str
    category: str
    row_number: int


@dataclass(frozen=True, slots=True)
class SequenceExample:
    session_id: str
    history: tuple[int, ...]
    target: int
    target_timestamp: str


def parse_timestamp(value: str) -> datetime:
    """Parse a YOOCHOOSE UTC timestamp into an aware datetime."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_clicks(path: str | Path) -> list[ClickEvent]:
    """Read yoochoose-clicks.dat without loading third-party libraries."""
    events: list[ClickEvent] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 4:
                raise ValueError(f"row {row_number}: expected >=4 columns, got {len(row)}")
            session_id, timestamp, item_id, category = row[:4]
            events.append(
                ClickEvent(
                    session_id=session_id,
                    timestamp=parse_timestamp(timestamp),
                    item_id=item_id,
                    category=category,
                    row_number=row_number,
                )
            )
    return events


def group_sessions(events: Iterable[ClickEvent]) -> dict[str, tuple[ClickEvent, ...]]:
    """Group events by session and sort chronologically with a stable tie-break."""
    grouped: dict[str, list[ClickEvent]] = defaultdict(list)
    for event in events:
        grouped[event.session_id].append(event)
    return {
        sid: tuple(sorted(items, key=lambda e: (e.timestamp, e.row_number)))
        for sid, items in grouped.items()
    }


def filter_sessions(
    sessions: dict[str, Sequence[ClickEvent]],
    *,
    min_session_length: int = 2,
    min_item_support: int = 5,
) -> dict[str, tuple[ClickEvent, ...]]:
    """Iteratively enforce session-length and item-support constraints.

    Iteration matters because removing low-support items can shorten sessions below
    the minimum length, and removing those sessions can in turn lower item support.
    """
    if min_session_length < 2:
        raise ValueError("min_session_length must be >= 2")
    if min_item_support < 1:
        raise ValueError("min_item_support must be >= 1")

    current = {sid: tuple(seq) for sid, seq in sessions.items() if len(seq) >= min_session_length}
    while True:
        support = Counter(event.item_id for seq in current.values() for event in seq)
        kept = {
            sid: tuple(event for event in seq if support[event.item_id] >= min_item_support)
            for sid, seq in current.items()
        }
        kept = {sid: seq for sid, seq in kept.items() if len(seq) >= min_session_length}
        if kept == current:
            return kept
        current = kept


def temporal_split(
    sessions: dict[str, Sequence[ClickEvent]],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, dict[str, tuple[ClickEvent, ...]]]:
    """Split whole sessions by session end time; never split a session across sets."""
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0 <= validation_fraction < 1):
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")

    ordered = sorted(
        sessions.items(),
        key=lambda kv: (kv[1][-1].timestamp, kv[0]),
    )
    n = len(ordered)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    return {
        "train": dict(ordered[:train_end]),
        "validation": dict(ordered[train_end:validation_end]),
        "test": dict(ordered[validation_end:]),
    }


def build_item_mapping(train_sessions: dict[str, Sequence[ClickEvent]]) -> dict[str, int]:
    """Build deterministic 1-based item IDs from training data only; 0 is reserved."""
    items = sorted({event.item_id for seq in train_sessions.values() for event in seq}, key=_natural_key)
    return {item_id: index for index, item_id in enumerate(items, start=1)}


def _natural_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def generate_examples(
    sessions: dict[str, Sequence[ClickEvent]],
    item_to_index: dict[str, int],
    *,
    max_history: int | None = None,
    require_fully_known_session: bool = True,
) -> list[SequenceExample]:
    """Generate prefix-to-next-item examples using a fixed training vocabulary.

    By default evaluation sessions containing unseen items are excluded entirely.
    This avoids silently deleting unknown events and creating artificial adjacency.
    """
    if max_history is not None and max_history < 1:
        raise ValueError("max_history must be >= 1 or None")

    result: list[SequenceExample] = []
    for session_id, seq in sorted(sessions.items()):
        if require_fully_known_session and any(event.item_id not in item_to_index for event in seq):
            continue
        encoded: list[tuple[int, ClickEvent]] = []
        for event in seq:
            index = item_to_index.get(event.item_id)
            if index is None:
                continue
            encoded.append((index, event))
        for target_pos in range(1, len(encoded)):
            start = 0 if max_history is None else max(0, target_pos - max_history)
            history = tuple(index for index, _ in encoded[start:target_pos])
            target, target_event = encoded[target_pos]
            result.append(
                SequenceExample(
                    session_id=session_id,
                    history=history,
                    target=target,
                    target_timestamp=target_event.timestamp.isoformat().replace("+00:00", "Z"),
                )
            )
    return result


def write_jsonl(examples: Iterable[SequenceExample], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), separators=(",", ":")) + "\n")
            count += 1
    return count


def preprocess_yoochoose(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    min_session_length: int = 2,
    min_item_support: int = 5,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    max_history: int | None = None,
) -> dict[str, object]:
    """Run the deterministic Milestone-2 preprocessing pipeline."""
    sessions = group_sessions(read_clicks(input_path))
    sessions = filter_sessions(
        sessions,
        min_session_length=min_session_length,
        min_item_support=min_item_support,
    )
    splits = temporal_split(
        sessions,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    item_to_index = build_item_mapping(splits["train"])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "item_mapping.json").write_text(
        json.dumps(item_to_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    example_counts: dict[str, int] = {}
    eligible_sessions: dict[str, int] = {}
    for split_name, split_sessions in splits.items():
        examples = generate_examples(split_sessions, item_to_index, max_history=max_history)
        example_counts[split_name] = write_jsonl(examples, out / f"{split_name}.jsonl")
        eligible_sessions[split_name] = len({example.session_id for example in examples})

    metadata: dict[str, object] = {
        "input": str(input_path),
        "session_count_after_filtering": len(sessions),
        "split_session_counts": {name: len(value) for name, value in splits.items()},
        "eligible_session_counts": eligible_sessions,
        "example_counts": example_counts,
        "train_item_count": len(item_to_index),
        "parameters": {
            "min_session_length": min_session_length,
            "min_item_support": min_item_support,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "max_history": max_history,
            "unknown_item_policy": "drop_entire_eval_session",
            "id_mapping_scope": "train_only",
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
