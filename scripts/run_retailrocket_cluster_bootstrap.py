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
from sequence_reclab.statistical_analysis import paired_cluster_bootstrap, write_dataclass_csv

PRIMARY_GAINS = {
    "context_gain": ("positionless_sasrec", "history_pool"),
    "positional_gain": ("sasrec", "positionless_sasrec"),
    "deep_seq_gain": ("sasrec", "markov"),
}
RETURNING_GAINS = {"recent_history_gain": ("history_pool", "bpr")}


def _cluster_id(example_id: str) -> str:
    source, separator, line_number = example_id.rpartition(":")
    if not separator or not source:
        raise ValueError(f"example_id lacks source/line separator: {example_id!r}")
    try:
        int(line_number)
    except ValueError as exc:
        raise ValueError(f"example_id lacks numeric line number: {example_id!r}") from exc
    return source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session-cluster paired bootstrap for the frozen Retailrocket gain decomposition."
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-lengths", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    args = parser.parse_args()

    mapping = json.loads((args.processed_dir / "item_mapping.json").read_text(encoding="utf-8"))
    item_count = len(mapping)
    train = load_examples_jsonl(args.processed_dir / "train.jsonl")
    validation = load_examples_jsonl(args.processed_dir / "validation.jsonl")
    primary_test = load_examples_jsonl(args.processed_dir / "test.jsonl")
    returning_test = load_examples_jsonl(args.processed_dir / "test_returning_users.jsonl")
    fixed_validation = fixed_evaluation_population(validation, args.history_lengths)
    fixed_primary_test = fixed_evaluation_population(primary_test, args.history_lengths)
    fixed_returning_test = fixed_evaluation_population(returning_test, args.history_lengths)
    sequences = reconstruct_sequences(train)
    config = TransformerRunConfig()
    intervals = []

    for history_length in sorted(args.history_lengths):
        train_window = truncate_examples(train, history_length)
        validation_window = truncate_examples(fixed_validation, history_length)
        cohort_specs = (
            ("primary", fixed_primary_test, PRIMARY_GAINS),
            ("returning_users", fixed_returning_test, RETURNING_GAINS),
        )
        for cohort, fixed_test, gain_definitions in cohort_specs:
            test_window = truncate_examples(fixed_test, history_length)
            cluster_ids = [_cluster_id(example.example_id) for example in test_window]
            required_models = sorted({model for pair in gain_definitions.values() for model in pair})
            deltas_by_gain = {gain: {} for gain in gain_definitions}

            for seed in args.seeds:
                _seed_python(seed)
                per_model = {}
                for model_name in required_models:
                    model = _fit_model(
                        model_name,
                        train_examples=train_window,
                        session_sequences=sequences,
                        item_count=item_count,
                        history_length=history_length,
                        seed=seed,
                        transformer=config,
                        transformer_validation_examples=(
                            validation_window
                            if model_name in {"positionless_sasrec", "sasrec"}
                            else None
                        ),
                    )
                    per_model[model_name] = evaluate_model_per_example(
                        model_name, model, test_window, item_count=item_count, policy=YOOCHOOSE_POLICY
                    )

                for gain, (high_name, low_name) in gain_definitions.items():
                    low_rows = per_model[low_name]
                    high_rows = per_model[high_name]
                    deltas_by_gain[gain][seed] = [
                        {metric: high[metric] - low[metric] for metric in high}
                        for low, high in zip(low_rows, high_rows, strict=True)
                    ]

            for gain, deltas_by_seed in deltas_by_gain.items():
                intervals.extend(
                    paired_cluster_bootstrap(
                        deltas_by_seed,
                        cluster_ids,
                        cohort=cohort,
                        gain=gain,
                        history_length=history_length,
                        bootstrap_samples=args.bootstrap_samples,
                        bootstrap_seed=args.bootstrap_seed + history_length,
                        base_bootstrap_seed=args.bootstrap_seed,
                    )
                )

    write_dataclass_csv(intervals, args.output)
    print(f"wrote {len(intervals)} cluster-bootstrap intervals to {args.output}")


if __name__ == "__main__":
    main()
