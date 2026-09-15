from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise ImportError(
        "Transformer models require PyTorch. Install SequenceRecLab with the 'deep' extra: "
        "python -m pip install -e '.[deep]'"
    ) from exc

from .models import _history_target


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    item_count: int
    max_history_length: int
    hidden_dim: int = 32
    num_heads: int = 2
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 256
    seed: int = 20260914

    def __post_init__(self) -> None:
        if self.item_count < 1:
            raise ValueError("item_count must be >= 1")
        if self.max_history_length < 1:
            raise ValueError("max_history_length must be >= 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("num_heads must divide hidden_dim")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


class _MatchedTransformerEncoder(nn.Module):
    """Shared encoder used by both sides of the positional ablation.

    Full-history self-attention is intentional. A causal mask would itself reveal
    token position, making the positionless control order-sensitive even with no
    positional embeddings. Both variants therefore receive the same unmasked
    observed prefix and use the same permutation-invariant masked-mean readout.
    """

    def __init__(self, config: TransformerConfig, *, use_positions: bool) -> None:
        super().__init__()
        self.config = config
        self.use_positions = use_positions
        self.item_embedding = nn.Embedding(config.item_count + 1, config.hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_history_length, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.output_bias = nn.Parameter(torch.zeros(config.item_count))

        if not use_positions:
            with torch.no_grad():
                self.position_embedding.weight.zero_()
            self.position_embedding.weight.requires_grad_(False)

    def encode(self, histories: torch.Tensor) -> torch.Tensor:
        if histories.ndim != 2:
            raise ValueError("histories must have shape [batch, sequence]")
        if histories.shape[1] > self.config.max_history_length:
            raise ValueError("history exceeds configured max_history_length; truncate in the data pipeline")
        if histories.shape[1] == 0:
            raise ValueError("histories must contain at least one token")

        valid = histories.ne(0)
        if not torch.all(valid.any(dim=1)):
            raise ValueError("each history must contain at least one non-padding item")

        states = self.item_embedding(histories)
        if self.use_positions:
            positions = torch.arange(histories.shape[1], device=histories.device)
            states = states + self.position_embedding(positions).unsqueeze(0)

        states = self.encoder(states, src_key_padding_mask=~valid)
        weights = valid.unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(dim=1) / weights.sum(dim=1)

    def forward(self, histories: torch.Tensor) -> torch.Tensor:
        representation = self.encode(histories)
        # Tied item embeddings ensure both variants share the same output
        # parameterization. Padding row zero is never a prediction candidate.
        return representation @ self.item_embedding.weight[1:].T + self.output_bias


class MatchedTransformerRecommender:
    """Training/scoring wrapper for the matched Transformer ablation."""

    use_positions: bool = False

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self.network = _MatchedTransformerEncoder(config, use_positions=self.use_positions)
        self._fitted = False

    def fit(self, examples: Iterable[object]) -> "MatchedTransformerRecommender":
        normalized = [_history_target(example) for example in examples]
        if not normalized:
            raise ValueError("at least one training example is required")
        torch.manual_seed(self.config.seed)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.network.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.network.train()
        generator = torch.Generator().manual_seed(self.config.seed)
        for _ in range(self.config.epochs):
            order = torch.randperm(len(normalized), generator=generator).tolist()
            for start in range(0, len(order), self.config.batch_size):
                batch = [normalized[index] for index in order[start : start + self.config.batch_size]]
                histories, targets = self._batch(batch)
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(histories)
                loss = nn.functional.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
        self.network.eval()
        self._fitted = True
        return self

    def score(self, history: Sequence[int], candidates: Iterable[int]) -> dict[int, float]:
        candidate_list = tuple(int(item) for item in candidates)
        self._validate_history(history)
        self._validate_candidates(candidate_list)
        tensor = self._history_tensor(history)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(tensor)[0]
        return {item: float(logits[item - 1]) for item in candidate_list}

    def representation(self, history: Sequence[int]) -> tuple[float, ...]:
        """Expose the pooled representation for invariance tests and diagnostics."""
        self._validate_history(history)
        tensor = self._history_tensor(history)
        self.network.eval()
        with torch.no_grad():
            vector = self.network.encode(tensor)[0]
        return tuple(float(value) for value in vector)

    def trainable_parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(parameter.shape)
            for name, parameter in self.network.named_parameters()
            if parameter.requires_grad
        }

    def all_parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(parameter.shape) for name, parameter in self.network.named_parameters()}

    def _batch(self, examples: Sequence[tuple[tuple[int, ...], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        for history, target in examples:
            self._validate_history(history)
            if not 1 <= target <= self.config.item_count:
                raise ValueError(f"target {target} is outside item vocabulary 1..{self.config.item_count}")
        max_length = max(len(history) for history, _ in examples)
        padded = torch.zeros((len(examples), max_length), dtype=torch.long)
        for row, (history, _) in enumerate(examples):
            padded[row, : len(history)] = torch.tensor(history, dtype=torch.long)
        targets = torch.tensor([target - 1 for _, target in examples], dtype=torch.long)
        return padded, targets

    def _history_tensor(self, history: Sequence[int]) -> torch.Tensor:
        return torch.tensor([tuple(int(item) for item in history)], dtype=torch.long)

    def _validate_history(self, history: Sequence[int]) -> None:
        if not history:
            raise ValueError("history must be non-empty")
        if len(history) > self.config.max_history_length:
            raise ValueError("history exceeds configured max_history_length; truncate in the data pipeline")
        for item in history:
            if not 1 <= int(item) <= self.config.item_count:
                raise ValueError(f"history item {item} is outside item vocabulary 1..{self.config.item_count}")

    def _validate_candidates(self, candidates: Sequence[int]) -> None:
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidates must be unique")
        for item in candidates:
            if not 1 <= item <= self.config.item_count:
                raise ValueError(f"candidate {item} is outside item vocabulary 1..{self.config.item_count}")


class PositionlessSASRec(MatchedTransformerRecommender):
    """Permutation-invariant Transformer control with positions disabled."""

    use_positions = False


class SASRec(MatchedTransformerRecommender):
    """Position-aware member of the matched next-item Transformer pair.

    SequenceRecLab uses full-prefix attention and masked-mean readout in both
    variants so explicit positional embeddings are the sole source of order
    information. This is a SASRec-style next-item Transformer ablation rather
    than a byte-for-byte reproduction of the original causal training recipe.
    """

    use_positions = True
