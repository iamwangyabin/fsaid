from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.prepare_synthbuster import OFFICIAL_GENERATORS, build_synthbuster_manifest


def _write_arrow_snapshot(root: Path, records: list[dict[str, object]]) -> None:
    pa = pytest.importorskip("pyarrow")
    root.mkdir(parents=True)
    filename = "data-00000-of-00001.arrow"
    table = pa.Table.from_pylist(records)
    with pa.OSFile(str(root / filename), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    (root / "state.json").write_text(
        json.dumps({"_data_files": [{"filename": filename}]}), encoding="utf-8"
    )


def test_builds_official_commercial_tools_view_without_writing_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    work = tmp_path / "work"
    official_csv = tmp_path / "commercial_tools.csv"
    manifest = tmp_path / "manifests" / "synthbuster.csv"

    source_paths = ["real_RAISE_1k/real.png", *[f"{key}/fake.png" for key in OFFICIAL_GENERATORS]]
    records = []
    for index, source_path in enumerate(source_paths):
        payload = f"image-{index}".encode()
        records.append(
            {
                "image_path": source_path,
                "md5": hashlib.md5(payload).hexdigest(),
                "width": 1,
                "height": 1,
                "image": payload,
            }
        )
    _write_arrow_snapshot(snapshot, records)
    before = sorted(path.relative_to(snapshot) for path in snapshot.rglob("*"))

    with official_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filename", "typ"))
        writer.writeheader()
        writer.writerow({"filename": "synthbuster/real_RAISE_1k/real.png", "typ": "real"})
        for source_generator in OFFICIAL_GENERATORS:
            writer.writerow(
                {
                    "filename": f"synthbuster/{source_generator}/fake.png",
                    "typ": source_generator,
                }
            )

    summary = build_synthbuster_manifest(
        snapshot,
        work,
        official_csv,
        manifest,
        expected_shards=1,
        expected_per_type=1,
        expected_path_set_sha256=None,
    )

    rows = list(csv.DictReader(manifest.open()))
    assert summary["official_unique_images"] == 5
    assert summary["manifest_rows"] == 8
    assert len(rows) == 8
    assert {row["generator"] for row in rows} == set(OFFICIAL_GENERATORS.values())
    assert all(sum(row["label"] == label for row in rows) == 4 for label in ("0", "1"))
    real_paths = [Path(row["path"]) for row in rows if row["label"] == "0"]
    assert all(os.path.samefile(real_paths[0], path) for path in real_paths[1:])
    assert sorted(path.relative_to(snapshot) for path in snapshot.rglob("*")) == before


def test_rejects_work_directory_inside_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    official_csv = tmp_path / "official.csv"
    official_csv.write_text("filename,typ\n", encoding="utf-8")

    with pytest.raises(ValueError, match="work_root"):
        build_synthbuster_manifest(
            snapshot,
            snapshot / "work",
            official_csv,
            tmp_path / "manifest.csv",
        )
