#!/usr/bin/env python3
"""Run the resumable GenImage FSD leave-one-out training campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genimage_arrow import GENIMAGE_ARROW_SHARDS, ensure_genimage_arrow_index
from train_fsd import GENIMAGE_CLASSES, train_fsd
from utils import verify_backends


DEFAULT_EXCLUDE_CLASSES = ("SD", "Midjourney", "glide", "ADM", "VQDM", "BigGAN")
_CHECKPOINT_PATTERN = re.compile(r"resnet50_step_(\d+)\.pth$")


def checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def latest_checkpoint(output_dir: Path, total_steps: int) -> Path | None:
    candidates = []
    for path in output_dir.glob("resnet50_step_*.pth"):
        step = checkpoint_step(path)
        if step is not None and step <= total_steps:
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_final_checkpoint(
    checkpoint: Path,
    exclude_class: str,
    total_steps: int,
    seed: int,
    workers: int,
    task_batch_size: int,
    accumulation_steps: int,
    pretrained_sha256: str,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("FSD campaign validation requires torch") from exc

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "step": total_steps,
        "exclude_class": exclude_class,
        "seed": seed,
        "workers": workers,
        "task_batch_size": task_batch_size,
        "accumulation_steps": accumulation_steps,
        "data_format": "arrow",
        "pretrained_sha256": pretrained_sha256,
    }
    actual = {"step": payload.get("step"), **payload.get("config", {})}
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"FSD final checkpoint validation failed: {mismatches}")
    return payload["config"]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _install_alias(alias: Path, checkpoint: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        if alias.resolve() != checkpoint.resolve():
            raise ValueError(f"FSD checkpoint alias points elsewhere: {alias}")
        return
    temporary = alias.with_name(f"{alias.name}.partial-{os.getpid()}")
    temporary.symlink_to(checkpoint.resolve())
    os.replace(temporary, alias)


def _load_download_summary(arrow_root: Path) -> dict[str, Any]:
    summary_path = arrow_root.resolve() / "download_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Verified complete GenImage download summary is missing: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("total_shards") != GENIMAGE_ARROW_SHARDS:
        raise ValueError(
            f"GenImage download summary reports {summary.get('total_shards')} shards, "
            f"expected {GENIMAGE_ARROW_SHARDS}"
        )
    if summary.get("existing_links", 0) + summary.get("downloaded_links", 0) != (
        GENIMAGE_ARROW_SHARDS
    ):
        raise ValueError("GenImage download summary does not cover every Arrow shard")
    return summary


def run_campaign(
    arrow_root: Path,
    arrow_index: Path,
    checkpoint_root: Path,
    code_commit: str,
    pretrained_checkpoint: Path,
    pretrained_sha256: str,
    exclude_classes: Sequence[str] = DEFAULT_EXCLUDE_CLASSES,
    device: str = "cuda:0",
    workers: int = 8,
    seed: int = 42,
    total_steps: int = 200_000,
    task_batch_size: int = 16,
    accumulation_steps: int = 1,
    save_interval: int = 10_000,
    log_interval: int = 1_000,
) -> list[dict[str, Any]]:
    verify_backends()
    if not code_commit.strip():
        raise ValueError("code_commit must be non-empty")
    pretrained_checkpoint = pretrained_checkpoint.expanduser().absolute()
    if not pretrained_checkpoint.is_file():
        raise FileNotFoundError(
            f"FSD pretrained checkpoint is missing: {pretrained_checkpoint}"
        )
    actual_pretrained_sha256 = sha256(pretrained_checkpoint)
    if actual_pretrained_sha256 != pretrained_sha256:
        raise ValueError(
            "FSD pretrained checkpoint SHA-256 mismatch: "
            f"expected {pretrained_sha256}, got {actual_pretrained_sha256}"
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not exclude_classes or len(set(exclude_classes)) != len(exclude_classes):
        raise ValueError("exclude_classes must be non-empty and unique")
    allowed = set(GENIMAGE_CLASSES) - {"real"}
    unknown = set(exclude_classes) - allowed
    if unknown:
        raise ValueError(f"Unknown GenImage exclude classes: {sorted(unknown)}")

    arrow_root = arrow_root.expanduser().resolve()
    arrow_index = arrow_index.expanduser().resolve()
    checkpoint_root = checkpoint_root.expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    download_summary = _load_download_summary(arrow_root)
    index_metadata = ensure_genimage_arrow_index(arrow_root, arrow_index)
    results = []

    for exclude_class in exclude_classes:
        output_dir = checkpoint_root / "training" / f"without_{exclude_class}"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_checkpoint = output_dir / f"resnet50_step_{total_steps}.pth"
        was_complete = final_checkpoint.is_file()
        resume_from = latest_checkpoint(output_dir, total_steps)
        print(
            json.dumps(
                {
                    "event": "campaign_start",
                    "exclude_class": exclude_class,
                    "resume_from": str(resume_from) if resume_from is not None else None,
                    "final_checkpoint": str(final_checkpoint),
                }
            ),
            flush=True,
        )
        if not was_complete:
            produced = train_fsd(
                arrow_root,
                output_dir,
                exclude_class,
                device=device,
                workers=workers,
                seed=seed,
                total_steps=total_steps,
                task_batch_size=task_batch_size,
                save_interval=save_interval,
                data_format="arrow",
                arrow_index=arrow_index,
                accumulation_steps=accumulation_steps,
                log_interval=log_interval,
                resume_from=resume_from,
                pretrained_checkpoint=pretrained_checkpoint,
            )
            if produced.resolve() != final_checkpoint.resolve():
                raise ValueError(
                    f"FSD training returned {produced}, expected {final_checkpoint}"
                )

        config = validate_final_checkpoint(
            final_checkpoint,
            exclude_class,
            total_steps,
            seed,
            workers,
            task_batch_size,
            accumulation_steps,
            pretrained_sha256,
        )
        checkpoint_hash = sha256(final_checkpoint)
        _install_alias(checkpoint_root / f"{exclude_class}.pth", final_checkpoint)
        if exclude_class == "SD":
            _install_alias(checkpoint_root / "source_without_opensdi.pth", final_checkpoint)

        summary_path = output_dir / "training_summary.json"
        if was_complete and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_summary = {
                "exclude_class": exclude_class,
                "checkpoint": str(final_checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "code_commit": code_commit,
                "pretrained_sha256": pretrained_sha256,
            }
            mismatches = {
                key: (summary.get(key), value)
                for key, value in expected_summary.items()
                if summary.get(key) != value
            }
            if mismatches:
                raise ValueError(f"FSD training summary validation failed: {mismatches}")
            event = "campaign_skip"
        else:
            summary = {
                "exclude_class": exclude_class,
                "checkpoint": str(final_checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "code_commit": code_commit,
                "pretrained_checkpoint": str(pretrained_checkpoint),
                "pretrained_sha256": pretrained_sha256,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "training_config": config,
                "arrow_revision": download_summary.get("revision"),
                "arrow_shards": len(index_metadata["shards"]),
                "arrow_index_counts": index_metadata["counts"],
            }
            _atomic_json(summary_path, summary)
            event = "campaign_complete"
        results.append(summary)
        print(json.dumps({"event": event, **summary}), flush=True)

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    _atomic_json(checkpoint_root / "campaign_summary.json", {"models": results})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-root", required=True, type=Path)
    parser.add_argument("--arrow-index", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True, type=Path)
    parser.add_argument("--pretrained-sha256", required=True)
    parser.add_argument(
        "--exclude-class",
        action="append",
        choices=GENIMAGE_CLASSES[1:],
        dest="exclude_classes",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_campaign(
        args.arrow_root,
        args.arrow_index,
        args.checkpoint_root,
        args.code_commit,
        args.pretrained_checkpoint,
        args.pretrained_sha256,
        exclude_classes=args.exclude_classes or DEFAULT_EXCLUDE_CLASSES,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        total_steps=args.total_steps,
        task_batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
