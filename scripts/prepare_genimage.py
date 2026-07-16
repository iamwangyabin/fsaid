#!/usr/bin/env python3
"""Prepare GenImage ImageFolder exports and benchmark manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_DIRECTORIES = {"nature": ("0_real", 0), "ai": ("1_fake", 1)}
ZIP_GENERATORS = {"biggan_imagenet": "BigGAN", "glide_imagenet": "glide"}
ORIGINAL_EIGHT_VIEW = {
    "Midjourney": "Midjourney",
    "stable_diffusion_v_1_4": "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5": "stable_diffusion_v_1_5",
    "wukong": "wukong",
    "ADM": "ADM",
    "glide": "glide",
    "VQDM": "VQDM",
    "BigGAN": "BigGAN",
}
PAPER_SIX_VIEW = {
    "Midjourney": "Midjourney",
    "stable_diffusion_v_1_4": "SD",
    "stable_diffusion_v_1_5": "SD",
    "wukong": "SD",
    "ADM": "ADM",
    "glide": "glide",
    "VQDM": "VQDM",
    "BigGAN": "BigGAN",
}
VIEWS = {"original-eight": ORIGINAL_EIGHT_VIEW, "paper-six": PAPER_SIX_VIEW}
PAPER_DATASET_ORDER = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")
OMNIDFA_ZERO_SHOT_VIEW = "omnidfa-zero-shot"
OMNIDFA_ZERO_SHOT_FAKE_SOURCES = (
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
)
OMNIDFA_ZERO_SHOT_REAL_SOURCE = "stable_diffusion_v_1_4"


def _class_images(class_root: Path, filesystem_order: bool) -> list[Path]:
    if not class_root.is_dir():
        raise FileNotFoundError(f"Missing GenImage class directory: {class_root}")
    candidates = class_root.iterdir() if filesystem_order else class_root.rglob("*")
    images = [
        path.resolve()
        for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not filesystem_order:
        images.sort()
    if not images:
        raise ValueError(f"No images found in {class_root}")
    return images


def build_omnidfa_zero_shot_manifest(
    dataset_root: Path,
    output_path: Path,
    filesystem_order: bool = False,
) -> dict[str, object]:
    """Build the aggregate GenImage view from OmniDFA's released class lists."""
    dataset_root = dataset_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    rows: list[dict[str, object]] = []
    counts = Counter()

    # The released evaluator processes the fake loader before the real loader.
    for source_generator in OMNIDFA_ZERO_SHOT_FAKE_SOURCES:
        images = _class_images(
            dataset_root / "test" / source_generator / "1_fake", filesystem_order
        )
        rows.extend(
            {
                "path": str(path),
                "label": 1,
                "generator": "GenImage",
                "split": "pool",
                "source_generator": source_generator,
            }
            for path in images
        )
        counts[(source_generator, 1)] += len(images)

    real_images = _class_images(
        dataset_root / "test" / OMNIDFA_ZERO_SHOT_REAL_SOURCE / "0_real",
        filesystem_order,
    )
    rows.extend(
        {
            "path": str(path),
            "label": 0,
            "generator": "GenImage",
            "split": "pool",
            "source_generator": OMNIDFA_ZERO_SHOT_REAL_SOURCE,
        }
        for path in real_images
    )
    counts[(OMNIDFA_ZERO_SHOT_REAL_SOURCE, 0)] += len(real_images)

    identities = [str(row["path"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("OmniDFA GenImage view contains duplicate absolute paths")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("path", "label", "generator", "split", "source_generator")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(dataset_root),
        "manifest": str(output_path),
        "view": OMNIDFA_ZERO_SHOT_VIEW,
        "rows": len(rows),
        "filesystem_order": filesystem_order,
        "official_fake_sources": list(OMNIDFA_ZERO_SHOT_FAKE_SOURCES),
        "official_real_source": OMNIDFA_ZERO_SHOT_REAL_SOURCE,
        "counts": {
            source_generator: {
                str(label): counts[(source_generator, label)]
                for label in (0, 1)
                if counts[(source_generator, label)]
            }
            for source_generator in OMNIDFA_ZERO_SHOT_FAKE_SOURCES
        },
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    output_path.with_suffix(".support.csv").unlink(missing_ok=True)
    return summary


def extract_zip_subsets(
    archive_path: Path,
    dataset_root: Path,
    generators: Iterable[str] = ("BigGAN", "glide"),
) -> dict[str, object]:
    archive_path = archive_path.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    wanted = set(generators)
    unknown = wanted - set(ZIP_GENERATORS.values())
    if unknown:
        raise ValueError(f"Unsupported ZIP generators: {sorted(unknown)}")

    counts: Counter[tuple[str, int]] = Counter()
    skipped_existing = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if len(parts) < 4 or parts[0].lower() != "test":
                continue
            generator = ZIP_GENERATORS.get(parts[1].lower())
            label_name = next(
                (part.lower() for part in parts[2:-1] if part.lower() in LABEL_DIRECTORIES),
                None,
            )
            label_info = LABEL_DIRECTORIES.get(label_name or "")
            if generator not in wanted or label_info is None:
                continue

            label_directory, label = label_info
            destination = dataset_root / "test" / generator / label_directory / parts[-1]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.stat().st_size == member.file_size:
                skipped_existing += 1
            else:
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            counts[(generator, label)] += 1

    missing = [
        f"{generator}/label_{label}"
        for generator in sorted(wanted)
        for label in (0, 1)
        if counts[(generator, label)] == 0
    ]
    if missing:
        raise ValueError("No matching ZIP images found for: " + ", ".join(missing))
    return {
        "archive": str(archive_path),
        "dataset_root": str(dataset_root),
        "extracted": sum(counts.values()),
        "skipped_existing": skipped_existing,
        "counts": {
            generator: {str(label): counts[(generator, label)] for label in (0, 1)}
            for generator in sorted(wanted)
        },
    }


def build_manifest(
    dataset_root: Path,
    output_path: Path,
    view: str,
    explicit_shots: int | None = None,
    seed: int = 40,
    filesystem_order: bool = False,
) -> dict[str, object]:
    if view == OMNIDFA_ZERO_SHOT_VIEW:
        if explicit_shots is not None:
            raise ValueError("OmniDFA zero-shot view does not use support rows")
        return build_omnidfa_zero_shot_manifest(
            dataset_root, output_path, filesystem_order=filesystem_order
        )

    dataset_root = dataset_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    generator_map = VIEWS[view]
    grouped: dict[str, dict[int, list[Path]]] = {}
    counts: Counter[tuple[str, int]] = Counter()

    for source_generator, target_generator in generator_map.items():
        generator_root = dataset_root / "test" / source_generator
        for directory_name, label in (("0_real", 0), ("1_fake", 1)):
            class_root = generator_root / directory_name
            images = _class_images(class_root, filesystem_order)
            grouped.setdefault(target_generator, {0: [], 1: []})[label].extend(images)
            counts[(target_generator, label)] += len(images)

    selected_support: set[Path] = set()
    if explicit_shots is not None:
        if explicit_shots <= 0:
            raise ValueError("explicit_shots must be positive")
        rng = random.Random(seed)
        generator_order = (
            PAPER_DATASET_ORDER
            if view == "paper-six"
            else tuple(dict.fromkeys(generator_map.values()))
        )
        for generator in generator_order:
            for label in (0, 1):
                candidates = grouped[generator][label]
                if len(candidates) <= explicit_shots:
                    raise ValueError(
                        f"{generator}/label_{label} needs more than {explicit_shots} images"
                    )
                selected_support.update(rng.sample(candidates, explicit_shots))

    rows: list[dict[str, object]] = []
    for generator, per_label in grouped.items():
        for label in (0, 1):
            default_split = "pool" if explicit_shots is None else "query"
            rows.extend(
                {
                    "path": str(path),
                    "label": label,
                    "generator": generator,
                    "split": "support" if path in selected_support else default_split,
                }
                for path in per_label[label]
            )

    identities = [str(row["path"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("GenImage export contains duplicate absolute paths")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "generator", "split"))
        writer.writeheader()
        writer.writerows(rows)

    support_path = output_path.with_suffix(".support.csv")
    if explicit_shots is not None:
        with support_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "label", "generator"))
            writer.writeheader()
            writer.writerows(
                {
                    "path": str(Path(str(row["path"])).relative_to(dataset_root)),
                    "label": row["label"],
                    "generator": row["generator"],
                }
                for row in rows
                if row["split"] == "support"
            )
    else:
        support_path.unlink(missing_ok=True)

    summary = {
        "dataset_root": str(dataset_root),
        "manifest": str(output_path),
        "support_manifest": str(support_path) if explicit_shots is not None else None,
        "view": view,
        "rows": len(rows),
        "explicit_shots": explicit_shots,
        "seed": seed if explicit_shots is not None else None,
        "filesystem_order": filesystem_order,
        "counts": {
            generator: {str(label): counts[(generator, label)] for label in (0, 1)}
            for generator in sorted(set(generator_map.values()))
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--view",
        choices=(*VIEWS, OMNIDFA_ZERO_SHOT_VIEW),
        default="paper-six",
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--extract-generators", nargs="+", default=["BigGAN", "glide"])
    parser.add_argument("--explicit-shots", type=int)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--filesystem-order", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result: dict[str, object] = {}
    if args.archive is not None:
        result["extraction"] = extract_zip_subsets(
            args.archive, args.dataset_root, args.extract_generators
        )
    result["manifest"] = build_manifest(
        args.dataset_root,
        args.output_manifest,
        args.view,
        explicit_shots=args.explicit_shots,
        seed=args.seed,
        filesystem_order=args.filesystem_order,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
