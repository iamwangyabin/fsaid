from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from utils import ConfigurationError


class FSDResNetEncoder:
    """The ResNet-50/1024 metric encoder used by FSD."""

    def __init__(self, device: str | torch.device, checkpoint: Path):
        try:
            import timm
            from torchvision import transforms
        except ImportError as exc:
            raise ConfigurationError("FSD requires timm and torchvision") from exc

        self.device = torch.device(device)
        # The released FSD evaluation code uses CenterCrop without a preceding resize.
        self.preprocess = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
        # The released checkpoint replaces the full state, so avoid a redundant download here.
        self.model = timm.create_model("resnet50", pretrained=False, num_classes=1024)
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
        state_dict = (
            payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor, fp16: bool = True) -> torch.Tensor:
        images = images.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=fp16 and self.device.type == "cuda",
            dtype=torch.float16,
        ):
            return self.model(images).float()


class CLIPLayerEncoder:
    """OpenAI CLIP ViT intermediate CLS feature used by FTNet.

    This executes the exact visual-token path up to the requested residual block
    and returns x[0], matching FTNet's released `layer_{n}_cls` extraction.
    """

    def __init__(
        self,
        device: str | torch.device,
        model_name: str = "ViT-L/14",
        layer: int = 12,
        download_root: str | None = None,
    ):
        try:
            import clip
        except ImportError as exc:
            raise ConfigurationError(
                "FTNet requires the pinned OpenAI CLIP dependency from environment.yml"
            ) from exc

        self.device = torch.device(device)
        self.layer = int(layer)
        self.model, self.preprocess = clip.load(
            model_name,
            device=self.device,
            jit=False,
            download_root=download_root,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        visual = self.model.visual
        if not hasattr(visual, "transformer") or not hasattr(visual, "conv1"):
            raise ConfigurationError("FTNet requires a CLIP Vision Transformer backbone")
        blocks = list(visual.transformer.resblocks.children())
        if not 1 <= self.layer <= len(blocks):
            raise ConfigurationError(f"clip_layer must be in [1, {len(blocks)}], got {self.layer}")

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        x = visual.conv1(images.to(self.device).type(self.model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        for block in list(visual.transformer.resblocks.children())[: self.layer]:
            x = block(x)
        return F.normalize(x[0].float(), dim=-1)


def load_image_batch(paths: Sequence[Path], preprocess, device: str | torch.device) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(preprocess(image.convert("RGB")))
    return torch.stack(images).to(device)
