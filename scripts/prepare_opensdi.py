#!/usr/bin/env python3
"""Materialize a read-only OpenSDI Parquet snapshot and freeze its few-shot split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


GENERATOR_FILES = {
    "sd15": "SD1.5",
    "sd2": "SD2.1",
    "sdxl": "SDXL",
    "sd3": "SD3",
    "flux": "FLUX.1",
}
GENERATOR_ORDER = tuple(GENERATOR_FILES.values())
LABEL_DIRECTORIES = {0: "0_real", 1: "1_fake"}


def _pyarrow_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "OpenSDI preparation requires pyarrow; install the project data extra"
        ) from exc
    return parquet


def _image_bytes(value: Any, snapshot_root: Path) -> bytes:
    if isinstance(value, dict):
        embedded = value.get("bytes")
        if isinstance(embedded, bytes):
            return embedded
        path_value = value.get("path")
        if isinstance(path_value, str) and path_value:
            source = Path(path_value).expanduser()
            if not source.is_absolute():
                source = snapshot_root / source
            return source.read_bytes()
    raise ValueError("OpenSDI image column does not contain bytes or a readable path")


def _suffix(source_key: str, image_value: Any) -> str:
    suffix = PurePosixPath(source_key).suffix.lower()
    if not suffix and isinstance(image_value, dict):
        path_value = image_value.get("path")
        if isinstance(path_value, str):
            suffix = PurePosixPath(path_value).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def build_opensdi_manifest(
    snapshot_root: Path,
    work_root: Path,
    output_path: Path,
    explicit_shots: int | None = 8,
    seed: int = 40,
    revision: str | None = None,
) -> dict[str, object]:
    if explicit_shots is not None and explicit_shots <= 0:
        raise ValueError("explicit_shots must be positive")

    parquet = _pyarrow_parquet()
    snapshot_root = snapshot_root.expanduser().resolve()
    work_root = work_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if work_root == snapshot_root or snapshot_root in work_root.parents:
        raise ValueError("work_root must be separate from the read-only snapshot")
    if output_path == snapshot_root or snapshot_root in output_path.parents:
        raise ValueError("output_manifest must be separate from the read-only snapshot")

    data_root = snapshot_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"OpenSDI snapshot is missing its data directory: {data_root}")

    grouped: dict[str, dict[int, list[dict[str, object]]]] = {
        generator: {0: [], 1: []} for generator in GENERATOR_ORDER
    }
    counts: Counter[tuple[str, int]] = Counter()
    written = 0
    reused = 0
    total_image_bytes = 0
    destinations: set[Path] = set()

    for source_name, generator in GENERATOR_FILES.items():
        shard_paths = sorted(data_root.glob(f"{source_name}-*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(f"No OpenSDI Parquet shards found for {source_name}")

        label_ordinals = {0: 0, 1: 0}
        for shard_path in shard_paths:
            parquet_file = parquet.ParquetFile(shard_path)
            missing = {"key", "image", "label"} - set(parquet_file.schema_arrow.names)
            if missing:
                raise ValueError(f"{shard_path} is missing columns: {sorted(missing)}")
            for batch in parquet_file.iter_batches(
                batch_size=128, columns=("key", "image", "label")
            ):
                for row in batch.to_pylist():
                    source_key = row["key"]
                    if not isinstance(source_key, str) or not source_key:
                        raise ValueError(f"{shard_path} contains an empty source key")
                    label = int(row["label"])
                    if label not in LABEL_DIRECTORIES:
                        raise ValueError(f"{shard_path} contains invalid label {label}")

                    payload = _image_bytes(row["image"], snapshot_root)
                    total_image_bytes += len(payload)
                    digest = hashlib.sha256(payload).hexdigest()
                    ordinal = label_ordinals[label]
                    label_ordinals[label] += 1
                    filename = f"{ordinal:05d}_{digest[:16]}{_suffix(source_key, row['image'])}"
                    destination = work_root / generator / LABEL_DIRECTORIES[label] / filename
                    if destination in destinations:
                        raise ValueError(f"Duplicate OpenSDI destination: {destination}")
                    destinations.add(destination)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.is_file() and destination.stat().st_size == len(payload):
                        reused += 1
                    else:
                        temporary = destination.with_name(destination.name + ".partial")
                        with temporary.open("wb") as handle:
                            handle.write(payload)
                        os.replace(temporary, destination)
                        written += 1

                    record: dict[str, object] = {
                        "path": str(destination),
                        "label": label,
                        "generator": generator,
                        "source_key": source_key,
                    }
                    grouped[generator][label].append(record)
                    counts[(generator, label)] += 1

    support_paths: set[str] = set()
    if explicit_shots is not None:
        rng = random.Random(seed)
        for generator in GENERATOR_ORDER:
            for label in (0, 1):
                candidates = grouped[generator][label]
                if len(candidates) <= explicit_shots:
                    raise ValueError(
                        f"{generator}/label_{label} needs more than {explicit_shots} images"
                    )
                support_paths.update(
                    str(record["path"]) for record in rng.sample(candidates, explicit_shots)
                )

    rows: list[dict[str, object]] = []
    for generator in GENERATOR_ORDER:
        for label in (0, 1):
            for record in grouped[generator][label]:
                default_split = "pool" if explicit_shots is None else "query"
                rows.append(
                    {
                        **record,
                        "split": (
                            "support"
                            if str(record["path"]) in support_paths
                            else default_split
                        ),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("path", "label", "generator", "split", "source_key")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    support_path = output_path.with_suffix(".support.csv")
    if explicit_shots is not None:
        with support_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("path", "label", "generator", "source_key")
            )
            writer.writeheader()
            writer.writerows(
                {
                    "path": str(Path(str(row["path"])).relative_to(work_root)),
                    "label": row["label"],
                    "generator": row["generator"],
                    "source_key": row["source_key"],
                }
                for row in rows
                if row["split"] == "support"
            )
    else:
        support_path.unlink(missing_ok=True)

    summary = {
        "snapshot_root": str(snapshot_root),
        "snapshot_revision": revision,
        "work_root": str(work_root),
        "manifest": str(output_path),
        "support_manifest": str(support_path) if explicit_shots is not None else None,
        "rows": len(rows),
        "explicit_shots": explicit_shots,
        "seed": seed if explicit_shots is not None else None,
        "written_images": written,
        "reused_images": reused,
        "image_bytes": total_image_bytes,
        "counts": {
            generator: {str(label): counts[(generator, label)] for label in (0, 1)}
            for generator in GENERATOR_ORDER
        },
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--explicit-shots", type=int, default=8)
    parser.add_argument(
        "--pool",
        action="store_true",
        help="write every row with split=pool instead of freezing support rows",
    )
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--revision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_opensdi_manifest(
        args.snapshot_root,
        args.work_root,
        args.output_manifest,
        explicit_shots=None if args.pool else args.explicit_shots,
        seed=args.seed,
        revision=args.revision,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
