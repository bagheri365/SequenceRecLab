#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sequence_reclab.experiments import (
    PRIMARY_YOOCHOOSE_MODELS,
    TransformerRunConfig,
    compute_gains,
    load_examples_jsonl,
    run_grid,
    write_results_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible SequenceRecLab experiment grid.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="yoochoose")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--history-lengths", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260914, 20260915, 20260916])
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_YOOCHOOSE_MODELS))
    parser.add_argument("--transformer-epochs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = json.loads((args.processed_dir / "item_mapping.json").read_text(encoding="utf-8"))
    train = load_examples_jsonl(args.processed_dir / "train.jsonl")
    evaluation = load_examples_jsonl(args.processed_dir / f"{args.split}.jsonl")

    rows = run_grid(
        dataset=args.dataset,
        train_examples=train,
        evaluation_examples=evaluation,
        item_count=len(mapping),
        history_lengths=args.history_lengths,
        seeds=args.seeds,
        models=args.models,
        split=args.split,
        transformer=TransformerRunConfig(epochs=args.transformer_epochs),
    )
    gains = compute_gains(rows)
    write_results_jsonl(rows, args.output_dir / "results.jsonl")
    write_results_jsonl(gains, args.output_dir / "gains.jsonl")
    print(f"wrote {len(rows)} result rows and {len(gains)} gain rows to {args.output_dir}")


if __name__ == "__main__":
    main()
