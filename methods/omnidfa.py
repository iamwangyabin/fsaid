from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from data import Sample
from methods.base import EvaluationOnlyMethod
from utils import ConfigurationError


class OmniValidationPreprocess:
    """Released p=0 JPEG/resize/blur validation pipeline before ToTensor."""

    def __init__(self, min_size: int = 256):
        self.min_size = int(min_size)
        self.resample_modes = (
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        is_jpeg = image.format == "JPEG"
        if image.mode == "P":
            image = image.convert("RGBA")
        image = image.convert("RGB")
        # RandomJPEG(p=0) still consumes an RNG draw for non-JPEG inputs.
        if not is_jpeg:
            random.random()
        width, height = image.size
        # CustomResizeKeepRatio(p=0) still draws before enforcing min_size.
        random.random()
        scale = max(1.0, self.min_size / width, self.min_size / height)
        if scale != 1.0:
            image = image.resize(
                (round(width * scale), round(height * scale)),
                resample=random.choice(self.resample_modes),
            )
        # RandomGaussianBlur(p=0) also consumes one Python RNG draw.
        random.random()
        return image


class SphereCenterLoss(nn.Module):
    def __init__(self, feat_dim: int, momentum: float = 0.99, normalize: bool = True):
        super().__init__()
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.normalize = normalize
        self.center_real = nn.Parameter(torch.randn(feat_dim))
        self.register_buffer("cosine_threshold", torch.tensor(1.0))


class TwinNeXt(nn.Module):
    """TwinNeXt architecture from the released OmniDFA detector."""

    def __init__(
        self,
        backbone_name: str = "convnext_small",
        mlp_hidden_dims: int = 512,
        out_dim: int = 128,
    ):
        super().__init__()
        try:
            from timm import create_model
            from timm.layers import Mlp
        except ImportError as exc:
            raise ConfigurationError("OmniDFA evaluation requires timm") from exc

        # The official detector checkpoint contains both complete backbone states.
        self.backbone1 = create_model(backbone_name, pretrained=False, num_classes=0)
        self.backbone2 = create_model(backbone_name, pretrained=False, num_classes=0)
        self.mlp = Mlp(
            in_features=self.backbone1.num_features + self.backbone2.num_features,
            hidden_features=mlp_hidden_dims,
            out_features=out_dim,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.backbone1(x1), self.backbone2(x2)], dim=1))


class OmniDFADetectionMethod(EvaluationOnlyMethod):
    """Evaluation-only real/fake branch of the released OmniDFA model."""

    name = "omnidfa_detection"
    reproduction_scope = "official_released_authenticity_inference"

    def __init__(self, config: dict[str, Any]):
        try:
            from torchvision import transforms
        except ImportError as exc:
            raise ConfigurationError("OmniDFA evaluation requires torchvision") from exc

        seed = int(config.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(config.get("device", "cuda:0"))
        self.batch_size = int(config.get("batch_size", 64))
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[str(config.get("dtype", "bfloat16"))]
        checkpoint = Path(str(config["checkpoint"])).expanduser().resolve()
        if not checkpoint.is_file():
            raise ConfigurationError(
                f"OmniDFA checkpoint does not exist: {checkpoint}. "
                "Download an official OmniDFA fold checkpoint before evaluation."
            )

        self.model = TwinNeXt("convnext_small", 512, 128)
        self.center = SphereCenterLoss(feat_dim=128)
        self._load_checkpoint(checkpoint)
        self.model.to(self.device).eval()
        self.center.to(self.device).eval()
        self.real_center = F.normalize(self.center.center_real.detach(), dim=0, p=2)
        # Official rule: cosine >= threshold is real. Negating both values
        # produces a fake-oriented score while preserving the exact decision.
        self.decision_threshold = -float(self.center.cosine_threshold.item())

        self.to_tensor = transforms.ToTensor()
        self.validation_preprocess = OmniValidationPreprocess(256)
        self.local_crop = transforms.RandomCrop(224)
        self.global_view = transforms.Compose([transforms.Resize(224), transforms.RandomCrop(224)])

    def _load_checkpoint(self, checkpoint: Path) -> None:
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except (AttributeError, ModuleNotFoundError, RuntimeError):
            import dill

            payload = torch.load(
                checkpoint,
                map_location="cpu",
                pickle_module=dill,
                weights_only=False,
            )
        if "model" not in payload or "cl_model" not in payload:
            raise ConfigurationError(
                "OmniDFA checkpoint must contain official 'model' and 'cl_model' states"
            )
        self.model.load_state_dict(payload["model"])
        self.center.load_state_dict(payload["cl_model"])

    def _views(self, samples: Sequence[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
        local, global_images = [], []
        for sample in samples:
            with Image.open(sample.path) as image:
                image = self.validation_preprocess(image)
                tensor = self.to_tensor(image)
            local.append(self.local_crop(tensor))
            global_images.append(self.global_view(tensor))
        return torch.stack(local).to(self.device), torch.stack(global_images).to(self.device)

    @torch.no_grad()
    def predict_fake_probability(self, generator: str, samples: Sequence[Sample]) -> list[float]:
        del generator
        scores = []
        for offset in range(0, len(samples), self.batch_size):
            local, global_images = self._views(samples[offset : offset + self.batch_size])
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.dtype != torch.float32 and self.device.type == "cuda",
                dtype=self.dtype,
            ):
                features = F.normalize(self.model(local, global_images), dim=-1, p=2)
                similarity_to_real = features @ self.real_center
            scores.extend((-similarity_to_real).float().cpu().tolist())
        return scores

    def official_metrics(
        self, labels: Sequence[int], fake_scores: Sequence[float]
    ) -> dict[str, float]:
        """Metrics used by OmniDFA's released authenticity evaluator.

        The official evaluator treats real as the positive class, averages
        real/fake accuracies, and uses a 20-threshold torchmetrics AP.
        """
        try:
            from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision
        except ImportError as exc:
            raise ConfigurationError("OmniDFA evaluation requires torchmetrics") from exc

        cosine = torch.tensor([-float(score) for score in fake_scores], dtype=torch.float32)
        real_targets = 1 - torch.tensor(labels, dtype=torch.int32)
        fake_mask = real_targets == 0
        real_mask = real_targets == 1
        threshold = float(self.center.cosine_threshold.item())
        accuracy = BinaryAccuracy(threshold=threshold)
        fake_accuracy = accuracy(cosine[fake_mask], real_targets[fake_mask])
        real_accuracy = accuracy(cosine[real_mask], real_targets[real_mask])
        average_precision = BinaryAveragePrecision(thresholds=20)(cosine, real_targets)
        return {
            "official_balanced_accuracy": float(((fake_accuracy + real_accuracy) / 2).item()),
            "official_real_positive_average_precision_20": float(average_precision.item()),
        }
