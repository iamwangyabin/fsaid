from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from data import Sample


class FewShotMethod(ABC):
    name: str
    decision_threshold: float = 0.5
    adaptation_mode: str = "few_shot"
    reproduction_scope: str = "algorithm_on_shared_protocol"

    @abstractmethod
    def adapt(
        self,
        generator: str,
        stage_support: Sequence[Sample],
        cumulative_support: Sequence[Sample],
        artifact_dir: Path,
    ) -> None:
        """Adapt with the labeled support available at a stage."""

    @abstractmethod
    def predict_fake_probability(self, generator: str, samples: Sequence[Sample]) -> list[float]:
        """Return a fake-oriented score for each sample, preserving input order."""

    def close(self) -> None:
        """Release method-specific resources."""

    def official_metrics(
        self, labels: Sequence[int], fake_scores: Sequence[float]
    ) -> dict[str, float]:
        """Return method-native metrics when they differ from the shared metrics."""
        del labels, fake_scores
        return {}


class EvaluationOnlyMethod(FewShotMethod):
    adaptation_mode = "evaluation_only"

    def adapt(
        self,
        generator: str,
        stage_support: Sequence[Sample],
        cumulative_support: Sequence[Sample],
        artifact_dir: Path,
    ) -> None:
        pass
