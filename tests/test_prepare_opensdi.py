import csv
from pathlib import Path

import pytest

from scripts.prepare_opensdi import GENERATOR_FILES, build_opensdi_manifest


def test_materializes_a_read_only_opensdi_snapshot(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    snapshot = tmp_path / "snapshot"
    data_root = snapshot / "data"
    data_root.mkdir(parents=True)

    for source_name in GENERATOR_FILES:
        table = pyarrow.table(
            {
                "key": [f"{source_name}-{index}.jpg" for index in range(4)],
                "image": pyarrow.array(
                    [
                        {"bytes": f"image-{source_name}-{index}".encode(), "path": None}
                        for index in range(4)
                    ],
                    type=pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())]),
                ),
                "label": [0, 0, 1, 1],
            }
        )
        parquet.write_table(table, data_root / f"{source_name}-00000.parquet")
    snapshot_contents = {
        path.relative_to(snapshot): path.read_bytes() for path in data_root.glob("*.parquet")
    }

    manifest = tmp_path / "opensdi.csv"
    summary = build_opensdi_manifest(
        snapshot,
        tmp_path / "work",
        manifest,
        explicit_shots=1,
        seed=40,
        revision="fixed",
    )

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with manifest.with_suffix(".support.csv").open(encoding="utf-8") as handle:
        support_rows = list(csv.DictReader(handle))

    assert summary["rows"] == 20
    assert summary["written_images"] == 20
    assert summary["counts"] == {
        generator: {"0": 2, "1": 2} for generator in GENERATOR_FILES.values()
    }
    assert sum(row["split"] == "support" for row in rows) == 10
    assert sum(row["split"] == "query" for row in rows) == 10
    assert len(support_rows) == 10
    assert all(not Path(row["path"]).is_absolute() for row in support_rows)
    assert all(Path(row["path"]).is_file() for row in rows)
    assert {
        path.relative_to(snapshot): path.read_bytes() for path in data_root.glob("*.parquet")
    } == snapshot_contents


def test_rejects_output_manifest_inside_snapshot(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(ValueError, match="output_manifest must be separate"):
        build_opensdi_manifest(
            snapshot,
            tmp_path / "work",
            snapshot / "manifest.csv",
        )

    assert not (snapshot / "manifest.csv").exists()


def test_can_write_a_pool_manifest_without_support_rows(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    snapshot = tmp_path / "snapshot"
    data_root = snapshot / "data"
    data_root.mkdir(parents=True)

    for source_name in GENERATOR_FILES:
        table = pyarrow.table(
            {
                "key": [f"{source_name}-real.jpg", f"{source_name}-fake.jpg"],
                "image": pyarrow.array(
                    [
                        {"bytes": f"real-{source_name}".encode(), "path": None},
                        {"bytes": f"fake-{source_name}".encode(), "path": None},
                    ],
                    type=pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())]),
                ),
                "label": [0, 1],
            }
        )
        parquet.write_table(table, data_root / f"{source_name}-00000.parquet")

    manifest = tmp_path / "opensdi.csv"
    summary = build_opensdi_manifest(
        snapshot,
        tmp_path / "work",
        manifest,
        explicit_shots=None,
    )

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"pool"}
    assert summary["support_manifest"] is None
    assert summary["seed"] is None
    assert not manifest.with_suffix(".support.csv").exists()
