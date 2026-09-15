from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from .evaluation import EvaluationPolicy, YOOCHOOSE_POLICY, build_candidates, evaluate_ranking, rank_items
from .models import BPRMatrixFactorization, FirstOrderMarkov, HistoryPool, PopularityModel

MODEL_NAMES = (
    "popularity",
    "markov",
    "bpr",
    "history_pool",
    "positionless_sasrec",
    "sasrec",
)

PRIMARY_YOOCHOOSE_MODELS = (
    "popularity",
    "markov",
    "history_pool",
    "positionless_sasrec",
    "sasrec",
)


@dataclass(frozen=True, slots=True)
class ExperimentExample:
    example_id: str
    history: tuple[int, ...]
    target: int
    identity_id: int | None = None


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    dataset: str
    split: str
    model: str
    seed: int
    history_length: int
    example_count: int
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class GainResult:
    dataset: str
    split: str
    seed: int
    history_length: int
    gain: str
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class TransformerRunConfig:
    hidden_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 256


def load_examples_jsonl(path: str | Path) -> list[ExperimentExample]:
    examples: list[ExperimentExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                history = tuple(int(item) for item in raw["history"])
                target = int(raw["target"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid history/target") from exc
            if not history:
                raise ValueError(f"{path}:{line_number}: history must be non-empty")
            identity_value = raw.get("user_id")
            identity_id = None if identity_value is None else int(identity_value)
            source_id = raw.get("session_id", raw.get("user_id", line_number))
            examples.append(
                ExperimentExample(
                    example_id=f"{source_id}:{line_number}",
                    history=history,
                    target=target,
                    identity_id=identity_id,
                )
            )
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def truncate_examples(
    examples: Iterable[ExperimentExample], history_length: int
) -> list[ExperimentExample]:
    if history_length < 1:
        raise ValueError("history_length must be >= 1")
    return [
        ExperimentExample(
            example_id=example.example_id,
            history=example.history[-history_length:],
            target=example.target,
            identity_id=example.identity_id,
        )
        for example in examples
    ]


def fixed_evaluation_population(
    examples: Iterable[ExperimentExample], history_lengths: Sequence[int]
) -> list[ExperimentExample]:
    lengths = _validate_history_lengths(history_lengths)
    minimum = max(lengths)
    eligible = [example for example in examples if len(example.history) >= minimum]
    if not eligible:
        raise ValueError(
            f"no evaluation examples have at least {minimum} prior interactions; "
            "reduce history_lengths or change the predeclared cohort"
        )
    return eligible


def reconstruct_sequences(examples: Iterable[ExperimentExample]) -> list[tuple[int, ...]]:
    """Recover each source sequence exactly once from ordered prefix examples.

    Preprocessing emits one example per target in source order, while each stored
    history may be capped by ``max_history``.  Taking only the longest stored
    prefix therefore loses early interactions for long sessions.  Instead, group
    examples by source, order them by their JSONL line number encoded in
    ``example_id``, seed the sequence from the first history, and append each
    successive target.
    """
    grouped: dict[str, list[tuple[int, ExperimentExample]]] = {}
    for example in examples:
        source, separator, line_number = example.example_id.rpartition(":")
        if not separator:
            raise ValueError(f"example_id lacks source/line separator: {example.example_id!r}")
        try:
            order = int(line_number)
        except ValueError as exc:
            raise ValueError(f"example_id lacks numeric line number: {example.example_id!r}") from exc
        grouped.setdefault(source, []).append((order, example))

    sequences: list[tuple[int, ...]] = []
    for source in sorted(grouped):
        ordered = sorted(grouped[source], key=lambda pair: pair[0])
        first = ordered[0][1]
        sequence = list(first.history)
        for _, example in ordered:
            if sequence[-len(example.history):] != list(example.history):
                raise ValueError(f"inconsistent prefix examples for source {source!r}")
            sequence.append(example.target)
        sequences.append(tuple(sequence))
    return sequences


def run_grid(
    *,
    dataset: str,
    train_examples: Sequence[ExperimentExample],
    evaluation_examples: Sequence[ExperimentExample],
    item_count: int,
    history_lengths: Sequence[int],
    seeds: Sequence[int],
    models: Sequence[str] = PRIMARY_YOOCHOOSE_MODELS,
    split: str = "test",
    policy: EvaluationPolicy = YOOCHOOSE_POLICY,
    transformer: TransformerRunConfig = TransformerRunConfig(),
) -> list[ExperimentResult]:
    lengths = _validate_history_lengths(history_lengths)
    normalized_models = _validate_models(models)
    normalized_seeds = _validate_seeds(seeds)
    if item_count < 1:
        raise ValueError("item_count must be >= 1")

    fixed_eval = fixed_evaluation_population(evaluation_examples, lengths)
    session_sequences = reconstruct_sequences(train_examples)
    results: list[ExperimentResult] = []

    for history_length in lengths:
        train_window = truncate_examples(train_examples, history_length)
        eval_window = truncate_examples(fixed_eval, history_length)
        for seed in normalized_seeds:
            _seed_python(seed)
            for model_name in normalized_models:
                model = _fit_model(
                    model_name,
                    train_examples=train_window,
                    session_sequences=session_sequences,
                    item_count=item_count,
                    history_length=history_length,
                    seed=seed,
                    transformer=transformer,
                )
                metrics = _evaluate_model(
                    model_name,
                    model,
                    eval_window,
                    item_count=item_count,
                    policy=policy,
                )
                results.append(
                    ExperimentResult(
                        dataset=dataset,
                        split=split,
                        model=model_name,
                        seed=seed,
                        history_length=history_length,
                        example_count=len(eval_window),
                        metrics=metrics,
                    )
                )
    return results


def compute_gains(results: Iterable[ExperimentResult]) -> list[GainResult]:
    grouped: dict[tuple[str, str, int, int], dict[str, ExperimentResult]] = {}
    for result in results:
        key = (result.dataset, result.split, result.seed, result.history_length)
        models = grouped.setdefault(key, {})
        if result.model in models:
            raise ValueError(f"duplicate result row for {key} model={result.model}")
        models[result.model] = result

    definitions = (
        ("recent_history_gain", "history_pool", "bpr"),
        ("context_gain", "positionless_sasrec", "history_pool"),
        ("positional_gain", "sasrec", "positionless_sasrec"),
        ("deep_seq_gain", "sasrec", "markov"),
    )
    gains: list[GainResult] = []
    for key, models in sorted(grouped.items()):
        for gain_name, high_name, low_name in definitions:
            if high_name not in models or low_name not in models:
                continue
            high = models[high_name]
            low = models[low_name]
            if high.example_count != low.example_count:
                raise ValueError(f"gain comparison uses mismatched populations for {key}: {high_name} vs {low_name}")
            metric_names = set(high.metrics) & set(low.metrics)
            gains.append(
                GainResult(
                    dataset=key[0],
                    split=key[1],
                    seed=key[2],
                    history_length=key[3],
                    gain=gain_name,
                    metrics={name: high.metrics[name] - low.metrics[name] for name in sorted(metric_names)},
                )
            )
    return gains


def write_results_jsonl(rows: Iterable[ExperimentResult | GainResult], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _fit_model(
    model_name: str,
    *,
    train_examples: Sequence[ExperimentExample],
    session_sequences: Sequence[Sequence[int]],
    item_count: int,
    history_length: int,
    seed: int,
    transformer: TransformerRunConfig,
):
    if model_name == "popularity":
        return PopularityModel().fit(session_sequences)
    if model_name == "markov":
        return FirstOrderMarkov().fit(session_sequences)
    if model_name == "history_pool":
        return HistoryPool().fit(train_examples)
    if model_name == "bpr":
        if any(example.identity_id is None for example in train_examples):
            raise ValueError("BPR experiments require persistent user_id values; session IDs are not substitutes")
        interactions = sorted(
            {
                (int(example.identity_id), item)
                for example in train_examples
                for item in (*example.history, example.target)
            }
        )
        return BPRMatrixFactorization(seed=seed).fit(interactions)
    if model_name in {"positionless_sasrec", "sasrec"}:
        from .transformers import PositionlessSASRec, SASRec, TransformerConfig

        config = TransformerConfig(
            item_count=item_count,
            max_history_length=history_length,
            hidden_dim=transformer.hidden_dim,
            num_heads=transformer.num_heads,
            num_layers=transformer.num_layers,
            dropout=transformer.dropout,
            learning_rate=transformer.learning_rate,
            weight_decay=transformer.weight_decay,
            epochs=transformer.epochs,
            batch_size=transformer.batch_size,
            seed=seed,
        )
        cls = PositionlessSASRec if model_name == "positionless_sasrec" else SASRec
        return cls(config).fit(train_examples)
    raise AssertionError(f"unhandled model {model_name}")


def _evaluate_model(
    model_name: str,
    model: object,
    examples: Sequence[ExperimentExample],
    *,
    item_count: int,
    policy: EvaluationPolicy,
) -> dict[str, float]:
    per_example: list[dict[str, float]] = []
    for example in examples:
        candidates = build_candidates(
            item_count=item_count,
            target=example.target,
            history=example.history,
            policy=policy,
            example_key=example.example_id,
        )
        if model_name == "popularity":
            scores = model.score(candidates)  # type: ignore[attr-defined]
        elif model_name == "bpr":
            if example.identity_id is None:
                raise ValueError("BPR evaluation requires persistent user_id values")
            scores = model.score(example.identity_id, candidates)  # type: ignore[attr-defined]
        else:
            scores = model.score(example.history, candidates)  # type: ignore[attr-defined]
        ranking = rank_items(scores, candidates)
        per_example.append(evaluate_ranking(ranking, example.target))

    metric_names = tuple(per_example[0])
    return {name: fmean(row[name] for row in per_example) for name in metric_names}


def _validate_history_lengths(history_lengths: Sequence[int]) -> tuple[int, ...]:
    lengths = tuple(int(value) for value in history_lengths)
    if not lengths or any(value < 1 for value in lengths):
        raise ValueError("history_lengths must contain positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("history_lengths must be unique")
    return tuple(sorted(lengths))


def _validate_models(models: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(models)
    if not normalized:
        raise ValueError("models must not be empty")
    unknown = sorted(set(normalized) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("models must be unique")
    return normalized


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("seeds must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique")
    return normalized


def _seed_python(seed: int) -> None:
    random.seed(seed)
