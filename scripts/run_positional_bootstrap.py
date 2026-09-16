#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sequence_reclab.evaluation import YOOCHOOSE_POLICY
from sequence_reclab.experiments import (
    TransformerRunConfig,
    _fit_model,
    _seed_python,
    evaluate_model_per_example,
    fixed_evaluation_population,
    load_examples_jsonl,
    reconstruct_sequences,
    truncate_examples,
)
from sequence_reclab.statistical_analysis import paired_example_bootstrap, write_dataclass_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired example bootstrap for SASRec - PositionlessSASRec.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-lengths", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    args = parser.parse_args()

    mapping = json.loads((args.processed_dir / "item_mapping.json").read_text(encoding="utf-8"))
    item_count = len(mapping)
    train = load_examples_jsonl(args.processed_dir / "train.jsonl")
    validation = load_examples_jsonl(args.processed_dir / "validation.jsonl")
    test = load_examples_jsonl(args.processed_dir / "test.jsonl")
    fixed_test = fixed_evaluation_population(test, args.history_lengths)
    fixed_validation = fixed_evaluation_population(validation, args.history_lengths)
    sequences = reconstruct_sequences(train)
    config = TransformerRunConfig()
    intervals = []

    for history_length in sorted(args.history_lengths):
        train_window = truncate_examples(train, history_length)
        validation_window = truncate_examples(fixed_validation, history_length)
        test_window = truncate_examples(fixed_test, history_length)
        deltas_by_seed = {}
        for seed in args.seeds:
            _seed_python(seed)
            per_model = {}
            for model_name in ("positionless_sasrec", "sasrec"):
                model = _fit_model(
                    model_name,
                    train_examples=train_window,
                    session_sequences=sequences,
                    item_count=item_count,
                    history_length=history_length,
                    seed=seed,
                    transformer=config,
                    transformer_validation_examples=validation_window,
                )
                per_model[model_name] = evaluate_model_per_example(
                    model_name, model, test_window, item_count=item_count, policy=YOOCHOOSE_POLICY
                )
            deltas_by_seed[seed] = [
                {metric: high[metric] - low[metric] for metric in high}
                for low, high in zip(per_model["positionless_sasrec"], per_model["sasrec"], strict=True)
            ]
        intervals.extend(
            paired_example_bootstrap(
                deltas_by_seed,
                history_length=history_length,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + history_length,
            )
        )
    write_dataclass_csv(intervals, args.output)
    print(f"wrote {len(intervals)} bootstrap intervals to {args.output}")


if __name__ == "__main__":
    main()
