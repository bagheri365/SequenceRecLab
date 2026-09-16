#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sequence_reclab.experiments import ExperimentResult, compute_gains
from sequence_reclab.statistical_analysis import load_jsonl, summarize_seed_rows, write_dataclass_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen multi-seed model results and derive matched gain rows."
    )
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_rows = []
    for path in args.results:
        raw_rows.extend(load_jsonl(path))
    model_rows = [ExperimentResult(**row) for row in raw_rows]
    gains = compute_gains(model_rows)
    summary_input = raw_rows + [
        {
            "dataset": row.dataset,
            "split": row.split,
            "seed": row.seed,
            "history_length": row.history_length,
            "gain": row.gain,
            "metrics": row.metrics,
        }
        for row in gains
    ]
    summaries = summarize_seed_rows(summary_input)
    write_dataclass_csv(summaries, args.output)
    print(f"wrote {len(summaries)} model/gain summary rows to {args.output}")


if __name__ == "__main__":
    main()
