#!/usr/bin/env python3
"""Download missing GenImage Arrow shards through an HF mirror without touching source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_REPO = "nebula/GenImage-arrow"
DEFAULT_REVISION = "93882903ce7dbd8723235d38ce23b5b1a5648ef4"
DEFAULT_ENDPOINT = "https://alpha.hf-mirror.com"
TOTAL_SHARDS = 1214


def _configure_hf_environment() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


def shard_name(index: int) -> str:
    return f"data-{index:05d}-of-{TOTAL_SHARDS:05d}.arrow"


def _assert_separate(existing_root: Path, download_root: Path, combined_root: Path) -> None:
    roots = [existing_root.resolve(), download_root.resolve(), combined_root.resolve()]
    labels = ["existing_root", "download_root", "combined_root"]
    for left_index, left in enumerate(roots):
        for right_index, right in enumerate(roots):
            if left_index == right_index:
                continue
            if left == right or left in right.parents:
                raise ValueError(
                    f"{labels[left_index]} must be separate from {labels[right_index]}"
                )


def find_missing_shards(existing_train_root: Path) -> list[str]:
    return [
        shard_name(index)
        for index in range(TOTAL_SHARDS)
        if not (existing_train_root / shard_name(index)).is_file()
    ]


def _repo_shard_metadata(
    repo_id: str, revision: str, endpoint: str
) -> dict[str, dict[str, object]]:
    _configure_hf_environment()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Download requires huggingface_hub") from exc

    info = HfApi(endpoint=endpoint).dataset_info(
        repo_id, revision=revision, files_metadata=True
    )
    if info.sha != revision:
        raise ValueError(f"HF mirror resolved {revision} to unexpected revision {info.sha}")

    metadata: dict[str, dict[str, object]] = {}
    for sibling in info.siblings:
        if not sibling.rfilename.startswith("train/data-"):
            continue
        lfs = sibling.lfs
        if sibling.size is None or lfs is None or not lfs.sha256:
            raise ValueError(f"Missing size or SHA-256 metadata for {sibling.rfilename}")
        metadata[Path(sibling.rfilename).name] = {
            "size": int(sibling.size),
            "sha256": str(lfs.sha256),
        }

    expected = {shard_name(index) for index in range(TOTAL_SHARDS)}
    missing_metadata = sorted(expected - set(metadata))
    if missing_metadata:
        raise ValueError(f"HF revision is missing shard metadata: {missing_metadata[:5]}")
    return metadata


def _validate_existing_sizes(
    existing_train_root: Path, metadata: dict[str, dict[str, object]]
) -> None:
    invalid = []
    for name, values in metadata.items():
        path = existing_train_root / name
        if path.is_file() and path.stat().st_size != values["size"]:
            invalid.append(name)
    if invalid:
        raise ValueError(f"Existing packaged shards have unexpected sizes: {invalid[:5]}")


def _download_missing(
    repo_id: str,
    revision: str,
    endpoint: str,
    download_root: Path,
    missing: list[str],
    workers: int,
    retries: int,
    retry_delay: float,
) -> None:
    _configure_hf_environment()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Download requires huggingface_hub") from exc

    def download_one(name: str) -> None:
        for attempt in range(1, retries + 2):
            try:
                hf_hub_download(
                    repo_id,
                    filename=f"train/{name}",
                    repo_type="dataset",
                    revision=revision,
                    endpoint=endpoint,
                    local_dir=download_root,
                )
                return
            except Exception as exc:  # noqa: BLE001 - resume the local partial shard
                if attempt > retries:
                    raise
                message = str(exc).splitlines()[0].split("?")[0][:240]
                print(
                    f"retry {attempt}/{retries} {name}: "
                    f"{type(exc).__name__}: {message}",
                    flush=True,
                )
                time.sleep(retry_delay * attempt)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, name): name for name in missing}
        for position, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - keep trying independent shards
                failures.append(name)
                message = str(exc).splitlines()[0].split("?")[0][:240]
                print(
                    f"failed {position}/{len(missing)} {name}: "
                    f"{type(exc).__name__}: {message}",
                    flush=True,
                )
            else:
                print(f"downloaded {position}/{len(missing)} {name}", flush=True)

    if failures:
        preview = ", ".join(sorted(failures)[:10])
        raise RuntimeError(
            f"Download round left {len(failures)} shards incomplete: {preview}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_downloaded_shards(
    download_train_root: Path,
    missing: list[str],
    metadata: dict[str, dict[str, object]],
) -> None:
    for position, name in enumerate(missing, start=1):
        path = download_train_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Downloaded shard is missing: {path}")
        expected_size = int(metadata[name]["size"])
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"Downloaded shard size mismatch for {name}: "
                f"expected {expected_size}, got {path.stat().st_size}"
            )
        actual_sha256 = _sha256(path)
        if actual_sha256 != metadata[name]["sha256"]:
            raise ValueError(
                f"Downloaded shard SHA-256 mismatch for {name}: "
                f"expected {metadata[name]['sha256']}, got {actual_sha256}"
            )
        print(f"verified {position}/{len(missing)} {name}", flush=True)


def build_combined_view(
    existing_train_root: Path,
    download_train_root: Path,
    combined_root: Path,
) -> tuple[int, int]:
    combined_train_root = combined_root / "train"
    combined_train_root.mkdir(parents=True, exist_ok=True)
    existing_links = 0
    downloaded_links = 0
    for index in range(TOTAL_SHARDS):
        name = shard_name(index)
        source = existing_train_root / name
        if source.is_file():
            existing_links += 1
        else:
            source = download_train_root / name
            downloaded_links += 1
        if not source.is_file():
            raise FileNotFoundError(f"Cannot build complete view; missing {name}")

        destination = combined_train_root / name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"Combined view has an unexpected symlink: {destination}")
            continue
        if destination.exists():
            raise ValueError(f"Combined view contains an unrelated file: {destination}")
        destination.symlink_to(source.resolve())
    return existing_links, downloaded_links


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-root", required=True, type=Path)
    parser.add_argument("--download-root", required=True, type=Path)
    parser.add_argument("--combined-root", required=True, type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.retries < 0 or args.retry_delay < 0:
        raise ValueError("retries and retry-delay must be non-negative")

    existing_root = args.existing_root.expanduser().resolve()
    download_root = args.download_root.expanduser().resolve()
    combined_root = args.combined_root.expanduser().resolve()
    _assert_separate(existing_root, download_root, combined_root)
    existing_train_root = existing_root / "train"
    if not existing_train_root.is_dir():
        raise FileNotFoundError(f"Existing GenImage train directory is missing: {existing_train_root}")
    download_root.mkdir(parents=True, exist_ok=True)
    combined_root.mkdir(parents=True, exist_ok=True)

    missing = find_missing_shards(existing_train_root)
    metadata = _repo_shard_metadata(args.repo_id, args.revision, args.endpoint)
    _validate_existing_sizes(existing_train_root, metadata)
    required_bytes = sum(int(metadata[name]["size"]) for name in missing)
    already_downloaded_bytes = sum(
        min((download_root / "train" / name).stat().st_size, int(metadata[name]["size"]))
        for name in missing
        if (download_root / "train" / name).is_file()
    )
    remaining_bytes = required_bytes - already_downloaded_bytes
    free_bytes = shutil.disk_usage(download_root).free
    reserve_bytes = 8 * 1024**3
    if free_bytes < remaining_bytes + reserve_bytes:
        raise OSError(
            f"Insufficient disk space: need {remaining_bytes + reserve_bytes} bytes "
            f"including reserve, have {free_bytes}"
        )

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "endpoint": args.endpoint,
                "existing_shards": TOTAL_SHARDS - len(missing),
                "missing_shards": len(missing),
                "download_bytes": required_bytes,
                "remaining_bytes": remaining_bytes,
                "free_bytes": free_bytes,
                "download_root": str(download_root),
                "combined_root": str(combined_root),
            },
            indent=2,
        ),
        flush=True,
    )

    if missing:
        _download_missing(
            args.repo_id,
            args.revision,
            args.endpoint,
            download_root,
            missing,
            args.workers,
            args.retries,
            args.retry_delay,
        )
        verify_downloaded_shards(download_root / "train", missing, metadata)

    existing_links, downloaded_links = build_combined_view(
        existing_train_root, download_root / "train", combined_root
    )
    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "endpoint": args.endpoint,
        "existing_root_read_only": str(existing_root),
        "download_root": str(download_root),
        "combined_root": str(combined_root),
        "total_shards": TOTAL_SHARDS,
        "existing_links": existing_links,
        "downloaded_links": downloaded_links,
        "downloaded_bytes": required_bytes,
        "downloaded_sha256": {
            name: metadata[name]["sha256"] for name in missing
        },
    }
    (combined_root / "download_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
