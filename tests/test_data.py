from pathlib import Path

import pytest
from PIL import Image

from data import load_manifest, scan_stage_folders, write_manifest
from utils import ConfigurationError


def test_scan_and_manifest_round_trip(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for generator in ("g1", "g2"):
        for class_name in ("real", "fake"):
            directory = dataset / generator / class_name
            directory.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(directory / "image.png")
    rows = scan_stage_folders(dataset)
    assert len(rows) == 4
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest, relative_to=tmp_path)
    loaded = load_manifest(manifest)
    assert [(row.path, row.label, row.generator, row.split) for row in rows] == [
        (row.path, row.label, row.generator, row.split) for row in loaded
    ]


def test_manifest_reports_truncated_rows_as_configuration_errors(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label,generator,split\nimage.png,1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="generator"):
        load_manifest(manifest)
