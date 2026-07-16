from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.download_genimage_train import (
    TOTAL_SHARDS,
    _assert_separate,
    _configure_hf_environment,
    build_combined_view,
    find_missing_shards,
    shard_name,
)


def test_configures_resumable_hf_timeouts_without_overriding_user_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "90")

    _configure_hf_environment()

    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "90"


def test_builds_complete_symlink_view_without_modifying_existing(tmp_path: Path) -> None:
    existing_root = tmp_path / "packaged" / "train"
    download_root = tmp_path / "download" / "train"
    combined_root = tmp_path / "combined"
    existing_root.mkdir(parents=True)
    download_root.mkdir(parents=True)

    split = TOTAL_SHARDS - 2
    for index in range(split):
        (existing_root / shard_name(index)).touch()
    for index in range(split, TOTAL_SHARDS):
        (download_root / shard_name(index)).touch()
    existing_before = sorted(path.name for path in existing_root.iterdir())

    assert find_missing_shards(existing_root) == [
        shard_name(TOTAL_SHARDS - 2),
        shard_name(TOTAL_SHARDS - 1),
    ]
    existing_links, downloaded_links = build_combined_view(
        existing_root, download_root, combined_root
    )

    assert existing_links == TOTAL_SHARDS - 2
    assert downloaded_links == 2
    assert len(list((combined_root / "train").iterdir())) == TOTAL_SHARDS
    assert (combined_root / "train" / shard_name(0)).resolve() == (
        existing_root / shard_name(0)
    ).resolve()
    assert (combined_root / "train" / shard_name(TOTAL_SHARDS - 1)).resolve() == (
        download_root / shard_name(TOTAL_SHARDS - 1)
    ).resolve()
    assert sorted(path.name for path in existing_root.iterdir()) == existing_before


def test_rejects_download_directory_inside_packaged_root(tmp_path: Path) -> None:
    existing = tmp_path / "packaged"
    with pytest.raises(ValueError, match="separate"):
        _assert_separate(existing, existing / "download", tmp_path / "combined")
