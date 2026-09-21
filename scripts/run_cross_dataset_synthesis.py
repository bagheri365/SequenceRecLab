#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sequence_reclab.synthesis import (
    build_publication_gain_rows,
    read_csv_rows,
    validate_estimand_boundaries,
    write_positional_replication_csv,
    write_publication_gain_csv,
    write_synthesis_markdown,
    write_manuscript_methods_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-ready cross-dataset gain tables from frozen analyses.")
    parser.add_argument("--yoochoose-summary", type=Path, required=True)
    parser.add_argument("--yoochoose-bootstrap", type=Path, required=True)
    parser.add_argument("--retailrocket-summary", type=Path, required=True)
    parser.add_argument("--retailrocket-bootstrap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    yoochoose = build_publication_gain_rows(
        "yoochoose",
        read_csv_rows(args.yoochoose_summary),
        uncertainty_rows=read_csv_rows(args.yoochoose_bootstrap),
        uncertainty_method="paired_example_bootstrap",
        uncertainty_gain="positional_gain",
    )
    retailrocket = build_publication_gain_rows(
        "retailrocket",
        read_csv_rows(args.retailrocket_summary),
        uncertainty_rows=read_csv_rows(args.retailrocket_bootstrap),
        uncertainty_method="paired_session_cluster_bootstrap",
    )
    rows = yoochoose + retailrocket
    validate_estimand_boundaries(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_publication_gain_csv(rows, args.output_dir / "gain_table.csv")
    write_positional_replication_csv(rows, args.output_dir / "positional_ndcg10.csv")
    write_synthesis_markdown(rows, args.output_dir / "synthesis.md")
    write_manuscript_methods_results(rows, args.output_dir / "manuscript_methods_results.md")
    print(f"wrote {len(rows)} gain rows and cross-dataset synthesis to {args.output_dir}")


if __name__ == "__main__":
    main()
