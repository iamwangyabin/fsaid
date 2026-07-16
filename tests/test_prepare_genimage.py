import csv
import zipfile
from pathlib import Path

from scripts.prepare_genimage import (
    OMNIDFA_ZERO_SHOT_FAKE_SOURCES,
    OMNIDFA_ZERO_SHOT_REAL_SOURCE,
    PAPER_SIX_VIEW,
    build_manifest,
    extract_zip_subsets,
)


def test_extracts_official_biggan_and_glide_directory_names(tmp_path: Path) -> None:
    archive_path = tmp_path / "genimage_test.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for source in ("biggan_imagenet", "glide_imagenet"):
            archive.writestr(f"test/{source}/nature/real.jpg", b"real")
            archive.writestr(f"test/{source}/ai/fake.png", b"fake")

    summary = extract_zip_subsets(archive_path, tmp_path / "GenImage")

    assert summary["extracted"] == 4
    assert (tmp_path / "GenImage/test/BigGAN/0_real/real.jpg").read_bytes() == b"real"
    assert (tmp_path / "GenImage/test/glide/1_fake/fake.png").read_bytes() == b"fake"


def test_builds_the_six_generator_paper_view(tmp_path: Path) -> None:
    dataset_root = tmp_path / "GenImage"
    for source_generator in PAPER_SIX_VIEW:
        for directory in ("0_real", "1_fake"):
            target = dataset_root / "test" / source_generator / directory
            target.mkdir(parents=True)
            (target / "image.png").write_bytes(b"image")

    manifest = tmp_path / "genimage.csv"
    summary = build_manifest(dataset_root, manifest, "paper-six")

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["rows"] == 16
    assert {row["generator"] for row in rows} == {
        "Midjourney",
        "SD",
        "ADM",
        "glide",
        "VQDM",
        "BigGAN",
    }
    assert sum(row["generator"] == "SD" for row in rows) == 6
    assert {row["split"] for row in rows} == {"pool"}


def test_can_freeze_the_released_python_sampling_loop(tmp_path: Path) -> None:
    dataset_root = tmp_path / "GenImage"
    for source_generator in PAPER_SIX_VIEW:
        for directory in ("0_real", "1_fake"):
            target = dataset_root / "test" / source_generator / directory
            target.mkdir(parents=True)
            for index in range(2):
                (target / f"image-{index}.png").write_bytes(b"image")

    manifest = tmp_path / "explicit.csv"
    build_manifest(
        dataset_root,
        manifest,
        "paper-six",
        explicit_shots=1,
        seed=40,
        filesystem_order=True,
    )

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with manifest.with_suffix(".support.csv").open(encoding="utf-8") as handle:
        support_rows = list(csv.DictReader(handle))
    assert sum(row["split"] == "support" for row in rows) == 12
    assert sum(row["split"] == "query" for row in rows) == 20
    assert len(support_rows) == 12
    assert all(not Path(row["path"]).is_absolute() for row in support_rows)


def test_builds_omnidfa_official_zero_shot_view(tmp_path: Path) -> None:
    dataset_root = tmp_path / "GenImage"
    for source_generator in OMNIDFA_ZERO_SHOT_FAKE_SOURCES:
        fake_root = dataset_root / "test" / source_generator / "1_fake"
        fake_root.mkdir(parents=True)
        (fake_root / "fake.png").write_bytes(b"fake")
    real_root = dataset_root / "test" / OMNIDFA_ZERO_SHOT_REAL_SOURCE / "0_real"
    real_root.mkdir(parents=True)
    (real_root / "real.png").write_bytes(b"real")

    manifest = tmp_path / "omnidfa.csv"
    summary = build_manifest(dataset_root, manifest, "omnidfa-zero-shot")

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["rows"] == len(OMNIDFA_ZERO_SHOT_FAKE_SOURCES) + 1
    assert [row["label"] for row in rows] == ["1"] * len(
        OMNIDFA_ZERO_SHOT_FAKE_SOURCES
    ) + ["0"]
    assert {row["generator"] for row in rows} == {"GenImage"}
    assert {row["split"] for row in rows} == {"pool"}
