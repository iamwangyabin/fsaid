from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import Sample
from methods.base import FewShotMethod
from models import CLIPLayerEncoder, load_image_batch


class LabeledImageDataset(Dataset):
    def __init__(self, paths: Sequence[Path], labels: Sequence[int], preprocess: Callable):
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have equal length")
        self.paths = tuple(paths)
        self.labels = tuple(int(label) for label in labels)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB")), self.labels[index]


def set_ftnet_seed(seed: int = 40) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_cache(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = features.t().contiguous()
    values = F.one_hot(labels, num_classes=num_classes).float().to(features.device)
    return keys, values


def cache_logits(
    features: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """FTNet Eq. 3-4: exp(-alpha * (1 - cosine affinity)) @ labels."""
    affinity = features @ cache_keys
    return torch.exp(-alpha * (1.0 - affinity)) @ cache_values


class FTNetMethod(FewShotMethod):
    """Training-free FTNet implemented inside the unified framework."""

    name = "ftnet"
    reproduction_scope = "paper_algorithm_unpublished_support_split"

    def __init__(self, config: dict[str, Any], train_keys: bool = False):
        set_ftnet_seed(40)
        self.train_keys = train_keys
        self.name = "ftnet_t" if train_keys else "ftnet"
        self.device = torch.device(config.get("device", "cuda"))
        self.batch_size = int(config.get("batch_size", 32 if train_keys else 64))
        self.num_workers = int(config.get("num_workers", 4 if train_keys else 0))
        self.alpha = float(config.get("alpha", 15.0))
        self.epochs = int(config.get("epochs", 20))
        self.learning_rate = float(config.get("learning_rate", 0.001))
        self.encoder = CLIPLayerEncoder(
            device=self.device,
            model_name=str(config.get("backbone", "ViT-L/14")),
            layer=int(config.get("clip_layer", 12)),
            download_root=config.get("download_root"),
        )
        self.cache_keys: torch.Tensor | None = None
        self.cache_values: torch.Tensor | None = None
        self.adapter: nn.Linear | None = None
        self.feature_positions: dict[str, int] | None = None
        self.cached_features: torch.Tensor | None = None

    def encode_samples(self, samples: Sequence[Sample]) -> torch.Tensor:
        outputs = []
        for offset in range(0, len(samples), self.batch_size):
            chunk = samples[offset : offset + self.batch_size]
            images = load_image_batch(
                [sample.path for sample in chunk], self.encoder.preprocess, self.device
            )
            outputs.append(self.encoder.encode(images))
        return torch.cat(outputs, dim=0)

    def set_feature_cache(
        self,
        identities: Sequence[str],
        features: torch.Tensor,
    ) -> None:
        if features.ndim != 2 or features.shape[0] != len(identities):
            raise ValueError("Feature cache dimensions do not match its identity list")
        if len(set(identities)) != len(identities):
            raise ValueError("Feature cache identity list contains duplicates")
        self.feature_positions = {
            identity: index for index, identity in enumerate(identities)
        }
        self.cached_features = features

    def _features(self, samples: Sequence[Sample]) -> torch.Tensor:
        if self.feature_positions is None or self.cached_features is None:
            return self.encode_samples(samples)
        try:
            indices = [self.feature_positions[sample.identity] for sample in samples]
        except KeyError as exc:
            raise ValueError(f"Feature cache is missing sample {exc.args[0]}") from exc
        return self.cached_features[indices].to(self.device)

    def _create_adapter(self) -> nn.Linear:
        if self.cache_keys is None:
            raise RuntimeError("Cache has not been built")
        adapter = nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(
            self.device
        )
        adapter.weight = nn.Parameter(self.cache_keys.t().clone())
        return adapter

    def _finetune_adapter(self, samples: Sequence[Sample]) -> nn.Linear:
        if self.cache_values is None:
            raise RuntimeError("Cache has not been built")
        set_ftnet_seed(40)
        dataset = LabeledImageDataset(
            [sample.path for sample in samples],
            [sample.label for sample in samples],
            self.encoder.preprocess,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        adapter = self._create_adapter()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.learning_rate, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * len(loader))
        for epoch in range(self.epochs):
            adapter.train()
            for images, labels in tqdm(
                loader,
                desc=f"FTNet-T epoch {epoch + 1}/{self.epochs}",
                leave=False,
            ):
                features = self.encoder.encode(images.to(self.device))
                affinity = adapter(features.to(adapter.weight.dtype))
                logits = torch.exp(-self.alpha * (1.0 - affinity)) @ self.cache_values
                loss = F.cross_entropy(logits, labels.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        return adapter.eval()

    def adapt(
        self,
        generator: str,
        stage_support: Sequence[Sample],
        cumulative_support: Sequence[Sample],
        artifact_dir: Path,
    ) -> None:
        del generator, stage_support
        features = self._features(cumulative_support)
        labels = torch.tensor(
            [sample.label for sample in cumulative_support],
            dtype=torch.long,
            device=self.device,
        )
        self.cache_keys, self.cache_values = build_cache(features, labels)
        if self.train_keys:
            self.adapter = self._finetune_adapter(cumulative_support)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.adapter.state_dict(), artifact_dir / "adapter.pt")
        else:
            self.adapter = None

    def predict_fake_probability(self, generator: str, samples: Sequence[Sample]) -> list[float]:
        del generator
        if self.cache_values is None:
            raise RuntimeError(f"{self.name} has not been adapted")
        features = self._features(samples)
        if self.train_keys:
            if self.adapter is None:
                raise RuntimeError("FTNet-T adapter is unavailable")
            affinity = self.adapter(features.to(self.adapter.weight.dtype))
            logits = torch.exp(-self.alpha * (1.0 - affinity)) @ self.cache_values
        else:
            if self.cache_keys is None:
                raise RuntimeError("FTNet cache is unavailable")
            logits = cache_logits(features, self.cache_keys, self.cache_values, self.alpha)
        return logits.softmax(dim=-1)[:, 1].detach().cpu().tolist()


class FTNetTMethod(FTNetMethod):
    reproduction_scope = "paper_settings_plus_historical_official_training_loop"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, train_keys=True)
