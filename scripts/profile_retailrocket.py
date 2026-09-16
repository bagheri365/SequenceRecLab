#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sequence_reclab.profiling import file_sha256
from sequence_reclab.retailrocket import (
    RETAILROCKET_EVENTS_SHA256,
    profile_retailrocket_sessions,
    read_view_events,
    sessionize_views,
)
from sequence_reclab.yoochoose import filter_sessions, temporal_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the frozen Retailrocket view-only 30-minute protocol.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inactivity-minutes", type=int, default=30)
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--min-item-support", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--history-lengths", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--skip-input-hash", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    digest = None if args.skip_input_hash else file_sha256(path)
    if digest is not None and digest != RETAILROCKET_EVENTS_SHA256:
        raise ValueError(f"Retailrocket events.csv SHA-256 mismatch: expected {RETAILROCKET_EVENTS_SHA256}, got {digest}")
    visitors = read_view_events(path)
    sessionized = sessionize_views(visitors, inactivity_minutes=args.inactivity_minutes)
    filtered = filter_sessions(sessionized, min_session_length=args.min_session_length, min_item_support=args.min_item_support)
    splits = temporal_split(filtered, train_fraction=args.train_fraction, validation_fraction=args.validation_fraction)
    profile = {
        "schema_version": 1,
        "dataset": "retailrocket",
        "protocol": {
            "primary_event": "view",
            "persistent_identity": "visitorid",
            "sessionization": "visitor view stream split after inactivity > threshold",
            "inactivity_minutes": args.inactivity_minutes,
            "repeats": "retained",
            "seen_item_policy": "allow",
            "item_metadata": "excluded",
            "secondary_events": ["addtocart", "transaction"],
            "history_lengths": args.history_lengths,
        },
        "input": {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest, "expected_sha256": RETAILROCKET_EVENTS_SHA256},
        "counts_before_filtering": {
            "view_visitor_count": len(visitors),
            "view_event_count": sum(len(value) for value in visitors.values()),
            "session_count": len(sessionized),
        },
        "filtered": profile_retailrocket_sessions(filtered, history_lengths=args.history_lengths),
        "splits": {name: profile_retailrocket_sessions(split, history_lengths=args.history_lengths) for name, split in splits.items()},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "profile.json"
    target.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"profile": str(target), "filtered_session_count": len(filtered)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
