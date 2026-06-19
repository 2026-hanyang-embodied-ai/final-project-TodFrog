"""Shared typed config records for supervised learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

ModelName: TypeAlias = Literal["mlp", "gru", "tcn", "transformer", "stgcn"]


@dataclass(frozen=True)
class ModelRunConfig:
    """One model/hyperparameter/seed run."""

    model: ModelName
    seed: int
    hidden_size: int
    dropout: float
    layers: int = 1
    heads: int = 2
    kernel_size: int = 3

    def run_id(self) -> str:
        """Return a stable run identifier."""

        return (
            f"{self.model}_seed{self.seed}_h{self.hidden_size}_"
            f"l{self.layers}_heads{self.heads}_k{self.kernel_size}_drop{self.dropout:g}"
        )
