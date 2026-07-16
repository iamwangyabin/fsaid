#!/usr/bin/env python3
"""Verify Hugging Face LFS files against cached repository metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(
    snapshot_root: Path,
    metadata_path: Path | None = None,
    expected_revision: str | None = None,
) -> dict[str, object]:
    snapshot_root = snapshot_root.expanduser().resolve()
    metadata_path = (
        metadata_path.expanduser().resolve()
        if metadata_path is not None
        else snapshot_root / ".hfd" / "repo_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    revision = metadata.get("sha")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(f"Snapshot revision is {revision!r}, expected {expected_revision!r}")

    failures: list[dict[str, object]] = []
    verified_bytes = 0
    verified_files = 0
    for sibling in metadata.get("siblings", []):
        lfs = sibling.get("lfs")
        if not isinstance(lfs, dict):
            continue
        relative_path = sibling.get("rfilename")
        if not isinstance(relative_path, str):
            failures.append({"path": relative_path, "reason": "invalid metadata path"})
            continue
        path = snapshot_root / relative_path
        expected_size = int(lfs["size"])
        expected_sha256 = str(lfs["sha256"])
        if not path.is_file():
            failures.append({"path": relative_path, "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            failures.append(
                {
                    "path": relative_path,
                    "reason": "size",
                    "expected": expected_size,
                    "actual": actual_size,
                }
            )
            continue
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            failures.append(
                {
                    "path": relative_path,
                    "reason": "sha256",
                    "expected": expected_sha256,
                    "actual": actual_sha256,
                }
            )
            continue
        verified_files += 1
        verified_bytes += actual_size

    if verified_files == 0 and not failures:
        raise ValueError("Metadata does not contain any LFS files")
    return {
        "snapshot_root": str(snapshot_root),
        "metadata": str(metadata_path),
        "revision": revision,
        "verified_lfs_files": verified_files,
        "verified_bytes": verified_bytes,
        "failures": failures,
        "ok": not failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--revision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_snapshot(args.snapshot_root, args.metadata, args.revision)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
