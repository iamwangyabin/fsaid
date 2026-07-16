from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

from utils import ConfigurationError


GENIMAGE_CLASSES = ("real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def train_fsd(
    data_root: Path,
    output_dir: Path,
    exclude_class: str,
    device: str = "cuda:0",
    workers: int = 8,
    seed: int = 42,
    total_steps: int = 200_000,
    task_batch_size: int = 16,
    save_interval: int = 10_000,
    data_format: str = "auto",
    arrow_index: Path | None = None,
    accumulation_steps: int = 1,
    log_interval: int = 1_000,
    resume_from: Path | None = None,
    pretrained_checkpoint: Path | None = None,
) -> Path:
    """Native FSD episodic training with the released architecture and schedule."""
    if exclude_class not in GENIMAGE_CLASSES or exclude_class == "real":
        raise ConfigurationError(
            f"exclude_class must be one of {GENIMAGE_CLASSES[1:]}, got {exclude_class!r}"
        )
    if workers < 0:
        raise ConfigurationError("workers must be non-negative")
    if total_steps <= 0:
        raise ConfigurationError("total_steps must be positive")
    if task_batch_size <= 0:
        raise ConfigurationError("task_batch_size must be positive")
    if save_interval <= 0:
        raise ConfigurationError("save_interval must be positive")
    if accumulation_steps <= 0:
        raise ConfigurationError("accumulation_steps must be positive")
    if log_interval <= 0:
        raise ConfigurationError("log_interval must be positive")
    if data_format not in {"auto", "image-folder", "arrow"}:
        raise ConfigurationError(
            "data_format must be one of: auto, image-folder, arrow"
        )
    if resume_from is not None:
        resume_from = resume_from.expanduser().resolve()
        if not resume_from.is_file():
            raise ConfigurationError(f"FSD resume checkpoint is missing: {resume_from}")
    pretrained_hash: str | None = None
    if pretrained_checkpoint is not None:
        pretrained_checkpoint = pretrained_checkpoint.expanduser().absolute()
        if not pretrained_checkpoint.is_file():
            raise ConfigurationError(
                f"FSD pretrained checkpoint is missing: {pretrained_checkpoint}"
            )
        pretrained_hash = _sha256(pretrained_checkpoint)

    try:
        import torch
        import timm
        from einops import rearrange
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        from methods.fsd import prototypical_loss
    except ImportError as exc:
        raise ConfigurationError("Install method dependencies before training FSD") from exc

    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    target_device = torch.device(device)

    class_names = [name for name in GENIMAGE_CLASSES if name != exclude_class]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    per_class_batch = (5 + 5) * task_batch_size
    resolved_data_format = data_format
    if resolved_data_format == "auto":
        resolved_data_format = (
            "arrow" if any((data_root.resolve() / "train").glob("*.arrow")) else "image-folder"
        )

    index_root: Path | None = None
    if resolved_data_format == "arrow":
        from genimage_arrow import ensure_genimage_arrow_index

        index_root = (
            arrow_index.resolve()
            if arrow_index is not None
            else data_root.resolve() / "fsd_index"
        )
        metadata = ensure_genimage_arrow_index(data_root, index_root)
        print(
            json.dumps(
                {
                    "data_format": "arrow",
                    "arrow_root": str(data_root.resolve()),
                    "arrow_index": str(index_root),
                    "arrow_shards": len(metadata["shards"]),
                    "arrow_counts": metadata["counts"],
                }
            ),
            flush=True,
        )

    def infinite_batches(class_name: str):
        if resolved_data_format == "arrow":
            from genimage_arrow import GenImageArrowClassDataset

            assert index_root is not None
            dataset = GenImageArrowClassDataset(
                data_root,
                index_root,
                class_name,
                "train",
                transform=transform,
            )
            dataset_path = data_root.resolve() / "train"
        else:
            dataset_path = data_root.resolve() / class_name / "train"
            if not dataset_path.is_dir():
                raise ConfigurationError(f"Missing FSD training directory: {dataset_path}")
            dataset = ImageFolder(dataset_path, transform=transform)
        if len(dataset) < per_class_batch:
            raise ConfigurationError(
                f"{dataset_path} needs at least {per_class_batch} images, found {len(dataset)}"
            )
        loader = DataLoader(
            dataset,
            batch_size=per_class_batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )
        while True:
            yield from loader

    streams = {name: iter(infinite_batches(name)) for name in class_names}
    model_options = {}
    if resume_from is None and pretrained_checkpoint is not None:
        model_options["pretrained_cfg_overlay"] = {"file": str(pretrained_checkpoint)}
    model = timm.create_model(
        "resnet50",
        pretrained=resume_from is None,
        num_classes=1024,
        **model_options,
    ).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=80_000)
    fp16 = target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 1
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=target_device, weights_only=False)
        saved_config = checkpoint.get("config", {})
        expected_resume_config = {
            "exclude_class": exclude_class,
            "seed": seed,
            "workers": workers,
            "task_batch_size": task_batch_size,
            "accumulation_steps": accumulation_steps,
            "data_format": resolved_data_format,
            "pretrained_sha256": pretrained_hash,
        }
        incompatible = {
            key: (saved_config.get(key), value)
            for key, value in expected_resume_config.items()
            if saved_config.get(key) != value
        }
        if incompatible:
            raise ConfigurationError(f"FSD resume configuration mismatch: {incompatible}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if fp16 and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        start_step = int(checkpoint["step"]) + 1
        print(
            json.dumps(
                {
                    "event": "resume",
                    "checkpoint": str(resume_from),
                    "completed_step": start_step - 1,
                    "next_step": start_step,
                }
            ),
            flush=True,
        )

    def save(step: int) -> Path:
        checkpoint = output_dir / f"resnet50_step_{step}.pth"
        temporary = checkpoint.with_name(checkpoint.name + ".partial")
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {
                    "exclude_class": exclude_class,
                    "seed": seed,
                    "workers": workers,
                    "total_steps": total_steps,
                    "save_interval": save_interval,
                    "log_interval": log_interval,
                    "task_batch_size": task_batch_size,
                    "accumulation_steps": accumulation_steps,
                    "effective_task_batch_size": task_batch_size * accumulation_steps,
                    "num_classes": 3,
                    "support": 5,
                    "query": 5,
                    "learning_rate": 1e-4,
                    "backbone": "resnet50",
                    "pretrained": True,
                    "pretrained_checkpoint": (
                        str(pretrained_checkpoint) if pretrained_checkpoint is not None else None
                    ),
                    "pretrained_sha256": pretrained_hash,
                    "data_format": resolved_data_format,
                    "arrow_index": str(index_root) if index_root is not None else None,
                },
                "python_random_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if fp16 else None,
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        return checkpoint

    if start_step > total_steps:
        assert resume_from is not None
        return resume_from

    last_checkpoint = resume_from or output_dir / "uninitialized.pth"
    started_at = time.monotonic()
    for step in range(start_step, total_steps + 1):
        model.train()
        optimizer.zero_grad()
        mean_loss = 0.0
        for _ in range(accumulation_steps):
            selected = random.sample(class_names, 3)
            batch = torch.stack([next(streams[name])[0] for name in selected], dim=0)
            batch = rearrange(batch.to(target_device), "n b c h w -> (n b) c h w")
            labels = torch.arange(3, device=target_device).repeat(task_batch_size * 5)
            with torch.autocast(device_type=target_device.type, enabled=fp16):
                embeddings = model(batch)
                embeddings = rearrange(
                    embeddings,
                    "(n b t) d -> b t n d",
                    n=3,
                    b=task_batch_size,
                )
                loss, _ = prototypical_loss(embeddings, labels, support_num=5)
            scaler.scale(loss / accumulation_steps).backward()
            mean_loss += loss.item() / accumulation_steps
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == start_step or step % log_interval == 0:
            elapsed = time.monotonic() - started_at
            completed = step - start_step + 1
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "total_steps": total_steps,
                        "loss": mean_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "elapsed_seconds": elapsed,
                        "seconds_per_step": elapsed / completed,
                    }
                ),
                flush=True,
            )
        if step % save_interval == 0 or step == total_steps:
            last_checkpoint = save(step)
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "step": step,
                        "path": str(last_checkpoint.resolve()),
                    }
                ),
                flush=True,
            )
    return last_checkpoint
