from __future__ import annotations

import random
from pathlib import Path

from utils import ConfigurationError


GENIMAGE_CLASSES = ("real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")


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

    def infinite_batches(class_name: str):
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
    model = timm.create_model("resnet50", pretrained=True, num_classes=1024).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=80_000)
    fp16 = target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save(step: int) -> Path:
        checkpoint = output_dir / f"resnet50_step_{step}.pth"
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
                    "task_batch_size": task_batch_size,
                    "num_classes": 3,
                    "support": 5,
                    "query": 5,
                    "learning_rate": 1e-4,
                },
            },
            checkpoint,
        )
        return checkpoint

    last_checkpoint = output_dir / "uninitialized.pth"
    for step in range(1, total_steps + 1):
        model.train()
        selected = random.sample(class_names, 3)
        batch = torch.stack([next(streams[name])[0] for name in selected], dim=0)
        batch = rearrange(batch.to(target_device), "n b c h w -> (n b) c h w")
        labels = torch.arange(3, device=target_device).repeat(task_batch_size * 5)
        optimizer.zero_grad()
        with torch.autocast(device_type=target_device.type, enabled=fp16):
            embeddings = model(batch)
            embeddings = rearrange(
                embeddings,
                "(n b t) d -> b t n d",
                n=3,
                b=task_batch_size,
            )
            loss, _ = prototypical_loss(embeddings, labels, support_num=5)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step % save_interval == 0 or step == total_steps:
            last_checkpoint = save(step)
    return last_checkpoint
