from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from PIL import Image

from data import Sample
from methods.base import EvaluationOnlyMethod
from utils import ConfigurationError


class ChannelLinear(nn.Linear):
    """ChannelLinear from the released GRIP-UNINA evaluation code."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        pool=None,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.compute_axis = 1
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axis_ref = len(x.shape) - 1
        x = torch.transpose(x, self.compute_axis, axis_ref)
        out_shape = list(x.shape)
        out_shape[-1] = self.out_features
        x = x.reshape(-1, x.shape[-1])
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None, :]
        x = torch.transpose(x.view(out_shape), axis_ref, self.compute_axis)
        if self.pool is not None:
            x = self.pool(x)
        return x


class OpenClipLinear(nn.Module):
    """Exact frozen OpenCLIP + linear head used by clipdet_latent10k(_plus)."""

    def __init__(self, device: torch.device, pretrained_path: str):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ConfigurationError("CLIPDet evaluation requires open_clip_torch") from exc

        backbone = open_clip.create_model("ViT-L-14", pretrained=pretrained_path)
        self.num_features = backbone.visual.proj.shape[0]
        backbone.visual.proj = None
        # The official implementation intentionally keeps the frozen backbone
        # outside the registered module tree; released checkpoints contain the head.
        self.bb = [backbone]
        self.normalize = True
        self.fc = ChannelLinear(self.num_features, 1)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        self.to(device)

    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        super().to(*args, **kwargs)
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.bb[0].eval()
            return self.bb[0].encode_image(x, normalize=self.normalize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))


class CLIPDetMethod(EvaluationOnlyMethod):
    """Evaluation-only integration of the released CLIP detector."""

    name = "clipdet"
    decision_threshold = 0.0
    reproduction_scope = "official_released_inference"

    def __init__(self, config: dict[str, Any]):
        try:
            from huggingface_hub import hf_hub_download
            from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize
            from torchvision.transforms import Resize, ToTensor
        except ImportError as exc:
            raise ConfigurationError(
                "CLIPDet evaluation requires huggingface_hub and torchvision"
            ) from exc

        self.device = torch.device(config.get("device", "cuda:0"))
        self.batch_size = int(config.get("batch_size", 32))
        checkpoint = Path(str(config["checkpoint"])).expanduser().resolve()
        if not checkpoint.is_file():
            raise ConfigurationError(
                f"CLIPDet head checkpoint does not exist: {checkpoint}. "
                "Obtain clipdet_latent10k_plus/weights.pth from the official repository."
            )

        backbone_path = config.get("backbone_checkpoint")
        if backbone_path is None:
            backbone_path = hf_hub_download(
                "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
                "open_clip_pytorch_model.bin",
            )
        self.model = OpenClipLinear(self.device, str(backbone_path))
        self._load_head(checkpoint)
        self.model.eval()
        self.transform = Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _load_head(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" in payload:
            state_dict = payload["model"]
            if any(key.startswith("module.") for key in state_dict):
                state_dict = {key[7:]: value for key, value in state_dict.items()}
        elif "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict_fake_probability(self, generator: str, samples: Sequence[Sample]) -> list[float]:
        del generator
        scores = []
        for offset in range(0, len(samples), self.batch_size):
            chunk = samples[offset : offset + self.batch_size]
            images = []
            for sample in chunk:
                with Image.open(sample.path) as image:
                    images.append(self.transform(image.convert("RGB")))
            logits = self.model(torch.stack(images).to(self.device))
            scores.extend(logits[:, 0].detach().cpu().tolist())
        return scores
