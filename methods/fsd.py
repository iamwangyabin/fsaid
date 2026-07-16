from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

from data import Sample
from methods.base import FewShotMethod
from models import FSDResNetEncoder, load_image_batch
from utils import ConfigurationError


def prototypical_scores(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Negative squared-Euclidean distance to mean class prototypes."""
    prototypes = support.mean(dim=1)
    return -((query[:, None, :] - prototypes[None, :, :]) ** 2).sum(dim=-1)


def prototypical_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, support_num: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact episodic FSD loss.

    embeddings: [task_batch, samples_per_class, class_count, feature_dim]
    """
    support = embeddings[:, :support_num]
    query = embeddings[:, support_num:]
    prototypes = support.mean(dim=1, keepdim=True)
    scores = -((rearrange(query, "b q n d -> b (q n) 1 d") - prototypes) ** 2).sum(dim=-1)
    scores = rearrange(scores, "b n c -> (b n) c")
    return F.cross_entropy(scores, labels), scores


class FSDMethod(FewShotMethod):
    """FSD implemented natively behind the common benchmark interface."""

    name = "fsd"
    reproduction_scope = "official_algorithm_on_fixed_shared_support"

    def __init__(self, config: dict[str, Any]):
        self.device = torch.device(config.get("device", "cuda"))
        self.fp16 = bool(config.get("fp16", True))
        self.batch_size = int(config.get("batch_size", 64))
        self.checkpoint_template = str(config["checkpoint"])
        self.encoder: FSDResNetEncoder | None = None
        self.loaded_checkpoint: Path | None = None
        self.prototypes: dict[str, torch.Tensor] = {}

    def _load_checkpoint(self, generator: str) -> None:
        checkpoint = Path(self.checkpoint_template.format(stage=generator)).expanduser().resolve()
        if checkpoint == self.loaded_checkpoint:
            return
        if not checkpoint.is_file():
            raise ConfigurationError(f"FSD checkpoint does not exist: {checkpoint}")
        self.encoder = FSDResNetEncoder(self.device, checkpoint)
        self.loaded_checkpoint = checkpoint

    def _features(self, samples: Sequence[Sample]) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("FSD has not loaded a checkpoint")
        outputs = []
        for offset in range(0, len(samples), self.batch_size):
            chunk = samples[offset : offset + self.batch_size]
            batch = load_image_batch(
                [sample.path for sample in chunk], self.encoder.preprocess, self.device
            )
            outputs.append(self.encoder.encode(batch, fp16=self.fp16))
        return torch.cat(outputs, dim=0)

    def adapt(
        self,
        generator: str,
        stage_support: Sequence[Sample],
        cumulative_support: Sequence[Sample],
        artifact_dir: Path,
    ) -> None:
        del cumulative_support, artifact_dir
        self._load_checkpoint(generator)
        per_class = []
        for label in (0, 1):
            selected = [sample for sample in stage_support if sample.label == label]
            if not selected:
                raise ConfigurationError(f"FSD support has no label {label} for {generator}")
            per_class.append(self._features(selected))
        self.prototypes[generator] = torch.stack(per_class, dim=0).mean(dim=1)

    def predict_fake_probability(self, generator: str, samples: Sequence[Sample]) -> list[float]:
        if generator not in self.prototypes:
            raise RuntimeError(f"FSD has no prototype for {generator}")
        query = self._features(samples)
        prototypes = self.prototypes[generator]
        scores = -((query[:, None, :] - prototypes[None, :, :]) ** 2).sum(dim=-1)
        return scores.softmax(dim=-1)[:, 1].detach().cpu().tolist()
