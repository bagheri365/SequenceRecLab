from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PublicationGainRow:
    dataset: str
    cohort: str
    gain: str
    history_length: int
    metric: str
    mean: float
    sample_sd: float
    n_seeds: int
    uncertainty_method: str
    lower_95: float | None
    upper_95: float | None


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def build_publication_gain_rows(
    dataset: str,
    summary_rows: Sequence[Mapping[str, str]],
    *,
    uncertainty_rows: Sequence[Mapping[str, str]] = (),
    uncertainty_method: str = "none",
    uncertainty_gain: str | None = None,
) -> list[PublicationGainRow]:
    intervals: dict[tuple[str, str, int, str], tuple[float, float]] = {}
    for row in uncertainty_rows:
        cohort = row.get("cohort", "primary")
        gain = row.get("gain", uncertainty_gain or "positional_gain")
        key = (cohort, gain, int(row["history_length"]), row["metric"])
        intervals[key] = (float(row["lower_95"]), float(row["upper_95"]))

    output: list[PublicationGainRow] = []
    for row in summary_rows:
        if row.get("kind") != "gain":
            continue
        cohort = row.get("cohort", "primary")
        gain = row["name"]
        history_length = int(row["history_length"])
        metric = row["metric"]
        interval = intervals.get((cohort, gain, history_length, metric))
        output.append(
            PublicationGainRow(
                dataset=dataset,
                cohort=cohort,
                gain=gain,
                history_length=history_length,
                metric=metric,
                mean=float(row["mean"]),
                sample_sd=float(row["sample_sd"]),
                n_seeds=int(row["n_seeds"]),
                uncertainty_method=uncertainty_method if interval is not None else "none",
                lower_95=interval[0] if interval is not None else None,
                upper_95=interval[1] if interval is not None else None,
            )
        )
    return sorted(output, key=lambda r: (r.dataset, r.cohort, r.gain, r.history_length, r.metric))


def validate_estimand_boundaries(rows: Iterable[PublicationGainRow]) -> None:
    for row in rows:
        if row.dataset == "yoochoose" and row.gain == "recent_history_gain":
            raise ValueError("YOOCHOOSE recent_history_gain is unavailable under the session-only protocol")
        if row.dataset == "retailrocket":
            expected = "returning_users" if row.gain == "recent_history_gain" else "primary"
            if row.cohort != expected:
                raise ValueError(
                    f"Retailrocket {row.gain} must use cohort={expected}, got {row.cohort}"
                )


def write_publication_gain_csv(rows: Sequence[PublicationGainRow], path: str | Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(PublicationGainRow.__dataclass_fields__)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def write_positional_replication_csv(rows: Sequence[PublicationGainRow], path: str | Path) -> None:
    selected = [r for r in rows if r.cohort == "primary" and r.gain == "positional_gain" and r.metric == "ndcg@10"]
    datasets = {r.dataset for r in selected}
    if datasets != {"yoochoose", "retailrocket"}:
        raise ValueError("positional replication table requires both yoochoose and retailrocket")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "history_length", "mean", "sample_sd", "n_seeds", "uncertainty_method", "lower_95", "upper_95"])
        writer.writeheader()
        for row in sorted(selected, key=lambda r: (r.dataset, r.history_length)):
            writer.writerow({
                "dataset": row.dataset,
                "history_length": row.history_length,
                "mean": row.mean,
                "sample_sd": row.sample_sd,
                "n_seeds": row.n_seeds,
                "uncertainty_method": row.uncertainty_method,
                "lower_95": row.lower_95,
                "upper_95": row.upper_95,
            })


def write_synthesis_markdown(rows: Sequence[PublicationGainRow], path: str | Path) -> None:
    positional = [r for r in rows if r.cohort == "primary" and r.gain == "positional_gain" and r.metric == "ndcg@10"]
    by_dataset = {dataset: sorted((r for r in positional if r.dataset == dataset), key=lambda r: r.history_length) for dataset in ("yoochoose", "retailrocket")}
    if any(len(values) != 3 for values in by_dataset.values()):
        raise ValueError("expected three primary positional NDCG@10 histories for each dataset")
    lines = [
        "# Cross-dataset synthesis",
        "",
        "## PositionalGain replication (NDCG@10)",
        "",
        "| Dataset | h2 | h3 | h5 | Uncertainty |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in ("yoochoose", "retailrocket"):
        values = by_dataset[dataset]
        method = values[0].uncertainty_method.replace("_", " ")
        cells = " | ".join(f"{r.mean:+.4f}" for r in values)
        lines.append(f"| {dataset} | {cells} | {method} |")
    lines += [
        "",
        "Both datasets estimate PositionalGain as SASRec - PositionlessSASRec on the primary cohort. Uncertainty methods differ by dataset and must not be presented as interchangeable: YOOCHOOSE uses a paired example-level percentile bootstrap, while Retailrocket uses a paired session-cluster percentile bootstrap. Training seeds are fixed in both analyses.",
        "",
        "Retailrocket RecentHistoryGain is estimated only on the returning-user cohort. It is a different estimand population from the primary-cohort ContextGain, PositionalGain, and DeepSeqGain and must not be added to them as a single decomposition. YOOCHOOSE RecentHistoryGain is unavailable because its session IDs are not persistent users.",
        "",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
