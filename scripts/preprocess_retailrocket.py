#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sequence_reclab.retailrocket import preprocess_retailrocket


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic Retailrocket view-session next-item splits.")
    parser.add_argument("--input", required=True, help="Path to Retailrocket events.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--inactivity-minutes", type=int, default=30)
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--min-item-support", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-history", type=int, default=None)
    parser.add_argument("--skip-input-hash", action="store_true")
    args = parser.parse_args()
    metadata = preprocess_retailrocket(
        args.input,
        args.output,
        inactivity_minutes=args.inactivity_minutes,
        min_session_length=args.min_session_length,
        min_item_support=args.min_item_support,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        max_history=args.max_history,
        expected_sha256=None if args.skip_input_hash else "3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
