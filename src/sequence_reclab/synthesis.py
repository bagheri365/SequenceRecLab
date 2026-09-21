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


def write_manuscript_methods_results(rows: Sequence[PublicationGainRow], path: str | Path) -> None:
    """Write a manuscript-ready Methods/Results/Limitations package from frozen rows."""
    validate_estimand_boundaries(rows)
    positional = [
        r for r in rows
        if r.cohort == "primary" and r.gain == "positional_gain" and r.metric == "ndcg@10"
    ]
    by_dataset = {
        dataset: sorted((r for r in positional if r.dataset == dataset), key=lambda r: r.history_length)
        for dataset in ("yoochoose", "retailrocket")
    }
    if any([r.history_length for r in values] != [2, 3, 5] for values in by_dataset.values()):
        raise ValueError("manuscript package requires h2/h3/h5 positional NDCG@10 for both datasets")
    if any(r.lower_95 is None or r.upper_95 is None for r in positional):
        raise ValueError("manuscript positional table requires frozen 95% intervals")

    rr_recent = sorted(
        (r for r in rows if r.dataset == "retailrocket" and r.cohort == "returning_users"
         and r.gain == "recent_history_gain" and r.metric == "ndcg@10"),
        key=lambda r: r.history_length,
    )
    if [r.history_length for r in rr_recent] != [2, 3, 5]:
        raise ValueError("manuscript package requires Retailrocket returning-user RecentHistoryGain h2/h3/h5")

    lines = [
        "# Manuscript-ready methods and results",
        "",
        "This reporting layer is downstream of the frozen experiments. It does not refit models, select checkpoints, change preprocessing, or use test results for tuning.",
        "",
        "## Methods",
        "",
        "### Tasks and evaluation",
        "",
        "We study next-item prediction on two implicit-feedback datasets. YOOCHOOSE uses session click sequences from the frozen 1/64-style subset; Retailrocket uses view events grouped into sessions by inactivity strictly greater than 30 minutes. Repeated items remain valid targets and previously seen items are not masked at evaluation. Both datasets use train-only item vocabularies, temporal train/validation/test splits, full-catalog evaluation, and primary history lengths 2, 3, and 5. Reported metrics are Recall@10/20, NDCG@10/20, and MRR@10.",
        "",
        "### Matched positional ablation",
        "",
        "PositionalGain is defined as SASRec - PositionlessSASRec. The two SASRec-style next-item Transformers share item embeddings, dimensionality, attention heads, depth, feed-forward width, objective, examples, candidate set, tied output parameterization, and masked-mean readout. Both use full self-attention over the observed history prefix; the positionless control fixes its same-shaped positional table to zero, whereas SASRec learns positional embeddings. This isolates the predictive value associated with explicit positional information within the matched architecture; it is not a claim of universal causal attribution.",
        "",
        "### Model selection and uncertainty",
        "",
        "Transformer checkpoints are selected only by validation full-catalog NDCG@10 with a maximum of 50 epochs, patience 5, and min_delta 0, after which the best validation checkpoint is restored for final test evaluation. Results use the three predeclared training seeds 17, 29, and 43. YOOCHOOSE PositionalGain intervals use a paired example-level percentile bootstrap. Retailrocket gain intervals use a paired session-cluster percentile bootstrap that resamples whole derived sessions. In both analyses the three training seeds are fixed rather than resampled, so these intervals quantify evaluation-population uncertainty conditional on those training runs, not population-of-training-seeds uncertainty.",
        "",
        "### Gain estimands",
        "",
        "The primary decomposition uses ContextGain = PositionlessSASRec - HistoryPool, PositionalGain = SASRec - PositionlessSASRec, and DeepSeqGain = SASRec - Markov. Retailrocket additionally estimates RecentHistoryGain = HistoryPool - BPR only on a separate returning-user cohort because BPR requires persistent users. YOOCHOOSE has no RecentHistoryGain estimand because its session IDs are not persistent user identities. Gains from different cohorts are not combined.",
        "",
        "## Results",
        "",
        "### Cross-dataset PositionalGain replication",
        "",
        "| Dataset | History | NDCG@10 gain | Seed SD | 95% interval | Uncertainty |",
        "|---|---:|---:|---:|---:|---|",
    ]
    labels = {
        "paired_example_bootstrap": "paired example bootstrap",
        "paired_session_cluster_bootstrap": "paired session-cluster bootstrap",
    }
    for dataset in ("yoochoose", "retailrocket"):
        for r in by_dataset[dataset]:
            lines.append(
                f"| {dataset} | {r.history_length} | {r.mean:+.4f} | {r.sample_sd:.4f} | "
                f"[{r.lower_95:+.4f}, {r.upper_95:+.4f}] | {labels.get(r.uncertainty_method, r.uncertainty_method)} |"
            )
    lines += [
        "",
        "Across both datasets, PositionalGain on NDCG@10 is positive at histories 2, 3, and 5 and increases with the available history. The magnitude is larger on YOOCHOOSE, while Retailrocket reproduces the same history-dependent pattern under a different dataset and sessionization protocol. Because the bootstrap sampling units differ, interval widths should not be compared as though they arose from the same uncertainty model.",
        "",
        "### Retailrocket returning-user RecentHistoryGain",
        "",
        "| History | NDCG@10 gain | Seed SD | 95% session-cluster interval |",
        "|---:|---:|---:|---:|",
    ]
    for r in rr_recent:
        lines.append(f"| {r.history_length} | {r.mean:+.4f} | {r.sample_sd:.4f} | [{r.lower_95:+.4f}, {r.upper_95:+.4f}] |")
    lines += [
        "",
        "RecentHistoryGain is reported separately because it is estimated on Retailrocket returning users rather than the primary sequential cohort. It is therefore not an additive component of the primary-cohort decomposition.",
        "",
        "## Limitations",
        "",
        "The study uses only three fixed training seeds, so seed means and sample standard deviations are descriptive and the bootstrap intervals do not represent uncertainty over arbitrary retraining runs. YOOCHOOSE uses example-level resampling, which does not preserve within-session dependence, whereas Retailrocket uses session-cluster resampling; the two interval procedures are intentionally labeled separately. RecentHistoryGain is available only for Retailrocket returning users and cannot be generalized automatically to its primary cohort or to YOOCHOOSE. Finally, the gain decomposition is a matched predictive ablation: it supports statements about incremental predictive value under these frozen architectures and protocols, not universal mechanistic or causal claims about sequence order.",
        "",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
