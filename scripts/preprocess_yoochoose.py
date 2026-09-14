#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sequence_reclab.yoochoose import preprocess_yoochoose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create deterministic next-item splits from YOOCHOOSE clicks.")
    parser.add_argument("--input", required=True, help="Path to yoochoose-clicks.dat")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--min-item-support", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-history", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = preprocess_yoochoose(
        args.input,
        args.output,
        min_session_length=args.min_session_length,
        min_item_support=args.min_item_support,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        max_history=args.max_history,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
