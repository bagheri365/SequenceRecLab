#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sequence_reclab.profiling import (
    assert_session_contract,
    assert_split_contract,
    file_sha256,
    profile_sessions,
    select_latest_session_fraction,
    write_session_manifest,
)
from sequence_reclab.yoochoose import filter_sessions, group_sessions, read_clicks, temporal_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile a deterministic YOOCHOOSE subset before model training.")
    parser.add_argument("--input", required=True, help="Path to yoochoose-clicks.dat")
    parser.add_argument("--output", required=True, help="Directory for profile.json and session_manifest.jsonl")
    parser.add_argument("--latest-session-fraction", type=float, default=1 / 64)
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--min-item-support", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--history-lengths", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("--skip-input-hash", action="store_true", help="Skip SHA-256 when a second full-file read is undesirable.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    raw_sessions = group_sessions(read_clicks(input_path))
    selected = select_latest_session_fraction(raw_sessions, args.latest_session_fraction)
    filtered = filter_sessions(
        selected,
        min_session_length=args.min_session_length,
        min_item_support=args.min_item_support,
    )
    assert_session_contract(
        filtered,
        min_session_length=args.min_session_length,
        min_item_support=args.min_item_support,
    )
    splits = temporal_split(
        filtered,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    assert_split_contract(splits)

    profile = {
        "schema_version": 1,
        "subset_definition": {
            "name": "sequence_reclab_latest_session_fraction",
            "fraction": args.latest_session_fraction,
            "selection_key": ["session_end_timestamp", "session_id"],
            "rounding": "ceil_with_minimum_one_session",
            "filtering_order": "select_latest_fraction_then_iterative_session_item_filter",
        },
        "parameters": {
            "min_session_length": args.min_session_length,
            "min_item_support": args.min_item_support,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "history_lengths": args.history_lengths,
        },
        "counts_before_filtering": {
            "raw_session_count": len(raw_sessions),
            "selected_session_count": len(selected),
        },
        "filtered_subset": profile_sessions(filtered, history_lengths=args.history_lengths),
        "splits": {
            name: profile_sessions(split, history_lengths=args.history_lengths)
            for name, split in splits.items()
        },
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": None if args.skip_input_hash else file_sha256(input_path),
        },
    }
    (output / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_count = write_session_manifest(splits, output / "session_manifest.jsonl")
    summary = {
        "profile": str(output / "profile.json"),
        "session_manifest": str(output / "session_manifest.jsonl"),
        "manifest_session_count": manifest_count,
        "filtered_session_count": len(filtered),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
