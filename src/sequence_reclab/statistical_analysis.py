from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SummaryRow:
    split: str
    cohort: str
    kind: str
    name: str
    history_length: int
    metric: str
    n_seeds: int
    mean: float
    sample_sd: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    history_length: int
    metric: str
    n_examples: int
    n_seeds: int
    bootstrap_samples: int
    base_bootstrap_seed: int
    effective_bootstrap_seed: int
    observed_mean_gain: float
    lower_95: float
    upper_95: float


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def summarize_seed_rows(rows: Iterable[Mapping[str, object]]) -> list[SummaryRow]:
    grouped: dict[tuple[str, str, str, str, int, str], list[float]] = {}
    for row in rows:
        kind, name = ("gain", str(row["gain"])) if "gain" in row else ("model", str(row["model"]))
        metrics = row["metrics"]
        if not isinstance(metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        for metric, value in metrics.items():
            key = (str(row["split"]), str(row.get("cohort", "primary")), kind, name, int(row["history_length"]), str(metric))
            grouped.setdefault(key, []).append(float(value))

    output: list[SummaryRow] = []
    for (split, cohort, kind, name, history_length, metric), values in sorted(grouped.items()):
        output.append(
            SummaryRow(
                split=split,
                cohort=cohort,
                kind=kind,
                name=name,
                history_length=history_length,
                metric=metric,
                n_seeds=len(values),
                mean=fmean(values),
                sample_sd=stdev(values) if len(values) > 1 else 0.0,
            )
        )
    return output


def paired_example_bootstrap(
    deltas_by_seed: Mapping[int, Sequence[Mapping[str, float]]],
    *,
    history_length: int,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260916,
    base_bootstrap_seed: int | None = None,
) -> list[BootstrapInterval]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    if not deltas_by_seed:
        raise ValueError("deltas_by_seed must not be empty")
    seeds = sorted(deltas_by_seed)
    n_examples = len(deltas_by_seed[seeds[0]])
    if n_examples < 1 or any(len(deltas_by_seed[seed]) != n_examples for seed in seeds):
        raise ValueError("all seeds must contain the same non-empty paired example population")
    metric_names = tuple(sorted(deltas_by_seed[seeds[0]][0]))
    if not metric_names:
        raise ValueError("paired deltas must contain metrics")
    for seed in seeds:
        for row in deltas_by_seed[seed]:
            if tuple(sorted(row)) != metric_names:
                raise ValueError("paired delta metric sets must match")

    effective_bootstrap_seed = bootstrap_seed
    if base_bootstrap_seed is None:
        base_bootstrap_seed = effective_bootstrap_seed
    rng = random.Random(effective_bootstrap_seed)
    distributions = {metric: [] for metric in metric_names}
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(n_examples) for _ in range(n_examples)]
        for metric in metric_names:
            seed_means = [
                fmean(float(deltas_by_seed[seed][index][metric]) for index in indices)
                for seed in seeds
            ]
            distributions[metric].append(fmean(seed_means))

    intervals: list[BootstrapInterval] = []
    for metric in metric_names:
        observed = fmean(
            fmean(float(row[metric]) for row in deltas_by_seed[seed]) for seed in seeds
        )
        ordered = sorted(distributions[metric])
        intervals.append(
            BootstrapInterval(
                history_length=history_length,
                metric=metric,
                n_examples=n_examples,
                n_seeds=len(seeds),
                bootstrap_samples=bootstrap_samples,
                base_bootstrap_seed=base_bootstrap_seed,
                effective_bootstrap_seed=effective_bootstrap_seed,
                observed_mean_gain=observed,
                lower_95=_percentile(ordered, 0.025),
                upper_95=_percentile(ordered, 0.975),
            )
        )
    return intervals


def write_dataclass_csv(rows: Iterable[object], path: str | Path) -> int:
    rows = list(rows)
    if not rows:
        raise ValueError("rows must not be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].__dataclass_fields__)  # type: ignore[attr-defined]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})
    return len(rows)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)

@dataclass(frozen=True, slots=True)
class ClusterBootstrapInterval:
    cohort: str
    gain: str
    history_length: int
    metric: str
    n_examples: int
    n_clusters: int
    n_seeds: int
    bootstrap_samples: int
    base_bootstrap_seed: int
    effective_bootstrap_seed: int
    observed_mean_gain: float
    lower_95: float
    upper_95: float


def paired_cluster_bootstrap(
    deltas_by_seed: Mapping[int, Sequence[Mapping[str, float]]],
    cluster_ids: Sequence[str],
    *,
    cohort: str,
    gain: str,
    history_length: int,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260916,
    base_bootstrap_seed: int | None = None,
) -> list[ClusterBootstrapInterval]:
    """Paired percentile bootstrap that resamples whole evaluation clusters.

    Training seeds are fixed. Each replicate draws cluster IDs with replacement;
    every example belonging to a sampled cluster is included, preserving within-
    session dependence and the model pairing on the identical resampled cohort.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    if not deltas_by_seed:
        raise ValueError("deltas_by_seed must not be empty")
    seeds = sorted(deltas_by_seed)
    n_examples = len(deltas_by_seed[seeds[0]])
    if n_examples < 1 or any(len(deltas_by_seed[seed]) != n_examples for seed in seeds):
        raise ValueError("all seeds must contain the same non-empty paired example population")
    if len(cluster_ids) != n_examples:
        raise ValueError("cluster_ids must align one-to-one with the paired example population")
    if any(not str(cluster_id) for cluster_id in cluster_ids):
        raise ValueError("cluster_ids must be non-empty")

    metric_names = tuple(sorted(deltas_by_seed[seeds[0]][0]))
    if not metric_names:
        raise ValueError("paired deltas must contain metrics")
    for seed in seeds:
        for row in deltas_by_seed[seed]:
            if tuple(sorted(row)) != metric_names:
                raise ValueError("paired delta metric sets must match")

    cluster_to_indices: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        cluster_to_indices.setdefault(str(cluster_id), []).append(index)
    clusters = tuple(sorted(cluster_to_indices))
    if not clusters:
        raise ValueError("at least one cluster is required")

    effective_bootstrap_seed = bootstrap_seed
    if base_bootstrap_seed is None:
        base_bootstrap_seed = effective_bootstrap_seed
    rng = random.Random(effective_bootstrap_seed)
    distributions = {metric: [] for metric in metric_names}
    for _ in range(bootstrap_samples):
        sampled_clusters = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        indices = [index for cluster_id in sampled_clusters for index in cluster_to_indices[cluster_id]]
        for metric in metric_names:
            seed_means = [
                fmean(float(deltas_by_seed[seed][index][metric]) for index in indices)
                for seed in seeds
            ]
            distributions[metric].append(fmean(seed_means))

    intervals: list[ClusterBootstrapInterval] = []
    for metric in metric_names:
        observed = fmean(
            fmean(float(row[metric]) for row in deltas_by_seed[seed]) for seed in seeds
        )
        ordered = sorted(distributions[metric])
        intervals.append(
            ClusterBootstrapInterval(
                cohort=cohort,
                gain=gain,
                history_length=history_length,
                metric=metric,
                n_examples=n_examples,
                n_clusters=len(clusters),
                n_seeds=len(seeds),
                bootstrap_samples=bootstrap_samples,
                base_bootstrap_seed=base_bootstrap_seed,
                effective_bootstrap_seed=effective_bootstrap_seed,
                observed_mean_gain=observed,
                lower_95=_percentile(ordered, 0.025),
                upper_95=_percentile(ordered, 0.975),
            )
        )
    return intervals
