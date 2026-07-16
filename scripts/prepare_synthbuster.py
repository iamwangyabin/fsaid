#!/usr/bin/env python3
"""Materialize the official CLIPDet SynthBuster evaluation from read-only Arrow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


OFFICIAL_GENERATORS = {
    "dalle2": "DALL-E 2",
    "dalle3": "DALL-E 3",
    "midjourney-v5": "Midjourney v5",
    "firefly": "Adobe Firefly",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
OFFICIAL_PATH_SET_SHA256 = "45ab1a5252079542757d10d5490eb57234dd70d6fccc906a2fdf5d04924af565"


def _pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "SynthBuster preparation requires pyarrow; install the project data extra"
        ) from exc
    return pa


def _read_official_csv(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    real_paths: list[str] = []
    fake_paths = {generator: [] for generator in OFFICIAL_GENERATORS}
    seen: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"filename", "typ"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Official CLIPDet CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            filename = str(row["filename"]).strip()
            typ = str(row["typ"]).strip()
            parts = PurePosixPath(filename).parts
            if parts and parts[0] == "synthbuster":
                parts = parts[1:]
            if not parts or any(part in {"", ".", ".."} for part in parts):
                raise ValueError(f"Invalid official path at line {line_number}: {filename!r}")
            source_path = PurePosixPath(*parts).as_posix()
            if PurePosixPath(source_path).suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image path at line {line_number}: {filename!r}")
            if source_path in seen:
                raise ValueError(f"Official CLIPDet CSV repeats {source_path!r}")
            seen.add(source_path)

            if typ == "real":
                if parts[0] != "real_RAISE_1k":
                    raise ValueError(f"Unexpected real path at line {line_number}: {filename!r}")
                real_paths.append(source_path)
            elif typ in OFFICIAL_GENERATORS:
                if parts[0] != typ:
                    raise ValueError(
                        f"Generator/path mismatch at line {line_number}: {typ!r}, {filename!r}"
                    )
                fake_paths[typ].append(source_path)
            else:
                raise ValueError(f"Unexpected official type at line {line_number}: {typ!r}")
    return real_paths, fake_paths


def _state_shards(snapshot_root: Path, expected_shards: int | None) -> list[Path]:
    state_path = snapshot_root / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"SynthBuster snapshot is missing {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    entries = state.get("_data_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("SynthBuster state.json has no Arrow data files")
    filenames = [entry.get("filename") for entry in entries if isinstance(entry, dict)]
    if len(filenames) != len(entries) or not all(isinstance(name, str) for name in filenames):
        raise ValueError("SynthBuster state.json contains an invalid data file entry")
    if len(set(filenames)) != len(filenames):
        raise ValueError("SynthBuster state.json repeats an Arrow data file")
    if expected_shards is not None and len(filenames) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} SynthBuster Arrow shards, found {len(filenames)}"
        )
    shards = [snapshot_root / str(name) for name in filenames]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SynthBuster snapshot is missing shards: {missing[:5]}")
    return shards


def _write_payload(destination: Path, payload: bytes) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == len(payload):
        return False
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
    os.replace(temporary, destination)
    return True


def _ensure_hardlink(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and os.path.samefile(source, destination):
            return False
        raise ValueError(f"Refusing to replace an unrelated SynthBuster work file: {destination}")
    os.link(source, destination)
    return True


def _materialize_wanted(
    snapshot_root: Path,
    work_root: Path,
    shards: list[Path],
    wanted: set[str],
) -> tuple[dict[str, Path], int, int, int]:
    pa = _pyarrow()
    materialized: dict[str, Path] = {}
    total_rows = 0
    written = 0
    reused = 0
    required_columns = {"image_path", "md5", "image"}

    for shard_path in shards:
        with pa.memory_map(str(shard_path), "r") as source:
            reader = pa.ipc.open_stream(source)
            missing = required_columns - set(reader.schema.names)
            if missing:
                raise ValueError(f"{shard_path} is missing columns: {sorted(missing)}")
            path_index = reader.schema.get_field_index("image_path")
            md5_index = reader.schema.get_field_index("md5")
            image_index = reader.schema.get_field_index("image")
            for batch in reader:
                paths = batch.column(path_index).to_pylist()
                digests = batch.column(md5_index).to_pylist()
                total_rows += len(batch)
                for row_index, source_path in enumerate(paths):
                    if source_path not in wanted:
                        continue
                    if source_path in materialized:
                        raise ValueError(f"Arrow snapshot repeats official image {source_path!r}")
                    image_value: Any = batch.column(image_index)[row_index].as_py()
                    if not isinstance(image_value, (bytes, bytearray, memoryview)):
                        raise ValueError(f"Arrow image payload is not binary for {source_path!r}")
                    payload = bytes(image_value)
                    expected_md5 = digests[row_index]
                    actual_md5 = hashlib.md5(payload).hexdigest()
                    if actual_md5 != expected_md5:
                        raise ValueError(
                            f"SynthBuster MD5 mismatch for {source_path}: "
                            f"expected {expected_md5}, got {actual_md5}"
                        )
                    destination = work_root / "official_unique" / Path(source_path)
                    if _write_payload(destination, payload):
                        written += 1
                    else:
                        reused += 1
                    materialized[source_path] = destination

    missing = sorted(wanted - set(materialized))
    if missing:
        raise ValueError(f"Official CLIPDet CSV images are absent from Arrow: {missing[:5]}")
    return materialized, total_rows, written, reused


def build_synthbuster_manifest(
    snapshot_root: Path,
    work_root: Path,
    official_csv: Path,
    output_path: Path,
    expected_shards: int | None = 31,
    expected_per_type: int | None = 1000,
    expected_path_set_sha256: str | None = OFFICIAL_PATH_SET_SHA256,
) -> dict[str, object]:
    snapshot_root = snapshot_root.expanduser().resolve()
    work_root = work_root.expanduser().resolve()
    official_csv = official_csv.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if work_root == snapshot_root or snapshot_root in work_root.parents:
        raise ValueError("work_root must be separate from the read-only Arrow snapshot")
    if output_path == snapshot_root or snapshot_root in output_path.parents:
        raise ValueError("output_manifest must be separate from the read-only Arrow snapshot")
    if not official_csv.is_file():
        raise FileNotFoundError(f"Official CLIPDet CSV does not exist: {official_csv}")

    real_paths, fake_paths = _read_official_csv(official_csv)
    counts = {"real": len(real_paths), **{key: len(value) for key, value in fake_paths.items()}}
    if expected_per_type is not None:
        unexpected = {key: value for key, value in counts.items() if value != expected_per_type}
        if unexpected:
            raise ValueError(
                f"Expected {expected_per_type} official images per type, found {unexpected}"
            )

    shards = _state_shards(snapshot_root, expected_shards)
    wanted = set(real_paths)
    for paths in fake_paths.values():
        wanted.update(paths)
    path_set_sha256 = hashlib.sha256("\n".join(sorted(wanted)).encode()).hexdigest()
    if (
        expected_path_set_sha256 is not None
        and path_set_sha256 != expected_path_set_sha256
    ):
        raise ValueError(
            "SynthBuster path set does not match CLIPDet's fixed official CSV: "
            f"expected {expected_path_set_sha256}, got {path_set_sha256}"
        )
    materialized, arrow_rows, written, reused = _materialize_wanted(
        snapshot_root, work_root, shards, wanted
    )

    rows: list[dict[str, object]] = []
    links_written = 0
    links_reused = 0
    for source_generator, display_generator in OFFICIAL_GENERATORS.items():
        for source_path in real_paths:
            canonical = materialized[source_path]
            destination = work_root / "evaluation" / source_generator / "0_real" / canonical.name
            if _ensure_hardlink(canonical, destination):
                links_written += 1
            else:
                links_reused += 1
            rows.append(
                {
                    "path": str(destination),
                    "label": 0,
                    "generator": display_generator,
                    "split": "pool",
                    "source_path": source_path,
                }
            )
        rows.extend(
            {
                "path": str(materialized[source_path]),
                "label": 1,
                "generator": display_generator,
                "split": "pool",
                "source_path": source_path,
            }
            for source_path in fake_paths[source_generator]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "label", "generator", "split", "source_path")
        )
        writer.writeheader()
        writer.writerows(rows)

    official_csv_sha256 = hashlib.sha256(official_csv.read_bytes()).hexdigest()
    summary = {
        "snapshot_root": str(snapshot_root),
        "work_root": str(work_root),
        "official_csv": str(official_csv),
        "official_csv_sha256": official_csv_sha256,
        "official_path_set_sha256": path_set_sha256,
        "manifest": str(output_path),
        "arrow_shards": len(shards),
        "arrow_rows": arrow_rows,
        "official_unique_images": len(wanted),
        "manifest_rows": len(rows),
        "written_images": written,
        "reused_images": reused,
        "written_real_hardlinks": links_written,
        "reused_real_hardlinks": links_reused,
        "counts": counts,
        "evaluation_counts": {
            display: {"0": len(real_paths), "1": len(fake_paths[source])}
            for source, display in OFFICIAL_GENERATORS.items()
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
    parser.add_argument("--official-csv", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--expected-shards", type=int, default=31)
    parser.add_argument("--expected-per-type", type=int, default=1000)
    parser.add_argument("--expected-path-set-sha256", default=OFFICIAL_PATH_SET_SHA256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_synthbuster_manifest(
        args.snapshot_root,
        args.work_root,
        args.official_csv,
        args.output_manifest,
        expected_shards=args.expected_shards,
        expected_per_type=args.expected_per_type,
        expected_path_set_sha256=args.expected_path_set_sha256,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
