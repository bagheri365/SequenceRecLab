from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class PopularityModel:
    """Global item-frequency baseline with deterministic scoring."""

    def __init__(self) -> None:
        self._counts: Counter[int] = Counter()

    def fit(self, item_sequences: Iterable[Sequence[int]]) -> "PopularityModel":
        counts: Counter[int] = Counter()
        for sequence in item_sequences:
            counts.update(sequence)
        self._counts = counts
        return self

    def score(self, candidates: Iterable[int]) -> dict[int, float]:
        return {item: float(self._counts[item]) for item in candidates}


class FirstOrderMarkov:
    """Maximum-likelihood first-order item transition baseline.

    Scores are conditional transition probabilities from the final history item.
    Items without an observed outgoing transition receive a popularity-backed
    probability. This keeps all training-vocabulary candidates rankable without
    injecting test information.
    """

    def __init__(self, *, fallback_weight: float = 1e-12) -> None:
        if fallback_weight < 0:
            raise ValueError("fallback_weight must be >= 0")
        self.fallback_weight = fallback_weight
        self._transitions: dict[int, Counter[int]] = {}
        self._outgoing_totals: Counter[int] = Counter()
        self._popularity: Counter[int] = Counter()
        self._total_items = 0

    def fit(self, item_sequences: Iterable[Sequence[int]]) -> "FirstOrderMarkov":
        transitions: dict[int, Counter[int]] = defaultdict(Counter)
        outgoing_totals: Counter[int] = Counter()
        popularity: Counter[int] = Counter()
        total_items = 0

        for sequence in item_sequences:
            if not sequence:
                continue
            popularity.update(sequence)
            total_items += len(sequence)
            for source, target in zip(sequence, sequence[1:]):
                transitions[source][target] += 1
                outgoing_totals[source] += 1

        self._transitions = dict(transitions)
        self._outgoing_totals = outgoing_totals
        self._popularity = popularity
        self._total_items = total_items
        return self

    def score(self, history: Sequence[int], candidates: Iterable[int]) -> dict[int, float]:
        candidate_list = tuple(candidates)
        if not history:
            return self._fallback_scores(candidate_list)

        source = history[-1]
        total = self._outgoing_totals[source]
        if total == 0:
            return self._fallback_scores(candidate_list)

        row = self._transitions.get(source, Counter())
        denominator = float(total)
        return {
            item: (row[item] / denominator) + self.fallback_weight * self._popularity_score(item)
            for item in candidate_list
        }

    def transition_count(self, source: int, target: int) -> int:
        return self._transitions.get(source, Counter())[target]

    def _fallback_scores(self, candidates: Sequence[int]) -> dict[int, float]:
        return {item: self._popularity_score(item) for item in candidates}

    def _popularity_score(self, item: int) -> float:
        if self._total_items == 0:
            return 0.0
        return self._popularity[item] / self._total_items


@dataclass(frozen=True, slots=True)
class BPRInteraction:
    user: int
    item: int


class BPRMatrixFactorization:
    """Small deterministic reference implementation of BPR-MF.

    This model requires persistent user IDs. It is intentionally not a YOOCHOOSE
    session model: held-out YOOCHOOSE sessions are new identities, so interpreting
    session factors as long-term personalization would be invalid.

    The implementation uses plain Python to keep the research contract testable
    without optional numerical dependencies. Large-scale experiments may swap in
    a numerically equivalent optimized backend later.
    """

    def __init__(
        self,
        *,
        factors: int = 16,
        learning_rate: float = 0.05,
        regularization: float = 0.0025,
        epochs: int = 20,
        seed: int = 20260914,
    ) -> None:
        if factors < 1:
            raise ValueError("factors must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.seed = seed
        self._user_factors: dict[int, list[float]] = {}
        self._item_factors: dict[int, list[float]] = {}
        self._user_items: dict[int, set[int]] = {}
        self._items: tuple[int, ...] = ()

    def fit(self, interactions: Iterable[BPRInteraction | tuple[int, int]]) -> "BPRMatrixFactorization":
        pairs = [
            (value.user, value.item) if isinstance(value, BPRInteraction) else (value[0], value[1])
            for value in interactions
        ]
        if not pairs:
            raise ValueError("at least one interaction is required")

        user_items: dict[int, set[int]] = defaultdict(set)
        items: set[int] = set()
        for user, item in pairs:
            user_items[user].add(item)
            items.add(item)
        item_list = tuple(sorted(items))
        eligible_users = tuple(sorted(user for user, seen in user_items.items() if len(seen) < len(item_list)))
        if not eligible_users:
            raise ValueError("BPR requires at least one user with an unobserved training item for negative sampling")

        rng = random.Random(self.seed)
        scale = 1.0 / math.sqrt(self.factors)
        user_factors = {
            user: [rng.uniform(-scale, scale) for _ in range(self.factors)]
            for user in sorted(user_items)
        }
        item_factors = {
            item: [rng.uniform(-scale, scale) for _ in range(self.factors)]
            for item in item_list
        }

        training_pairs = [(user, item) for user, item in pairs if user in eligible_users]
        for _ in range(self.epochs):
            rng.shuffle(training_pairs)
            for user, positive in training_pairs:
                seen = user_items[user]
                negative = rng.choice(item_list)
                while negative in seen:
                    negative = rng.choice(item_list)
                self._update(user_factors[user], item_factors[positive], item_factors[negative])

        self._user_factors = user_factors
        self._item_factors = item_factors
        self._user_items = {user: set(seen) for user, seen in user_items.items()}
        self._items = item_list
        return self

    def score(self, user: int, candidates: Iterable[int]) -> dict[int, float]:
        if user not in self._user_factors:
            raise KeyError(
                f"unseen user {user}: BPR-MF requires a persistent identity observed in training; "
                "do not substitute a held-out session ID"
            )
        user_vector = self._user_factors[user]
        scores: dict[int, float] = {}
        for item in candidates:
            if item not in self._item_factors:
                raise KeyError(f"item {item} is outside the BPR training vocabulary")
            scores[item] = _dot(user_vector, self._item_factors[item])
        return scores

    def observed_items(self, user: int) -> frozenset[int]:
        return frozenset(self._user_items.get(user, set()))

    def _update(self, user: list[float], positive: list[float], negative: list[float]) -> None:
        x_uij = _dot(user, positive) - _dot(user, negative)
        # sigmoid(-x) is d[-log(sigmoid(x))]/dx in magnitude, evaluated stably.
        gradient = _sigmoid_negative(x_uij)
        lr = self.learning_rate
        reg = self.regularization
        for index in range(self.factors):
            u = user[index]
            p = positive[index]
            n = negative[index]
            user[index] += lr * (gradient * (p - n) - reg * u)
            positive[index] += lr * (gradient * u - reg * p)
            negative[index] += lr * (-gradient * u - reg * n)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sigmoid_negative(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return exp_neg / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return 1.0 / (1.0 + exp_pos)
