from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from genimage_arrow import (
    GenImageArrowClassDataset,
    build_genimage_arrow_index,
    classify_genimage_arrow_path,
)


@pytest.mark.parametrize(
    ("image_path", "expected"),
    [
        ("ADM/train/ai/a.png", ("train", "ADM")),
        ("Glide/val/ai/a.png", ("val", "glide")),
        ("stable_diffusion_v_1_4/train/ai/a.png", ("train", "SD")),
        ("stable_diffusion_v_1_5/train/ai/a.png", ("train", "SD")),
        ("wukong/train/ai/a.png", ("train", "SD")),
        ("stable_diffusion_v_1_4/train/nature/a.jpg", ("train", "real")),
        ("ADM/train/nature/a.jpg", None),
    ],
)
def test_classifies_official_fsd_logical_views(
    image_path: str, expected: tuple[str, str] | None
) -> None:
    assert classify_genimage_arrow_path(image_path) == expected


def _png_bytes() -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buffer = BytesIO()
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_arrow(path: Path, image_paths: list[str]) -> None:
    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    payload = _png_bytes()
    table = pa.table(
        {
            "image_path": image_paths,
            "md5": ["unused"] * len(image_paths),
            "width": [4] * len(image_paths),
            "height": [3] * len(image_paths),
            "image": [payload] * len(image_paths),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def test_builds_compact_index_and_reads_image_payload(tmp_path: Path) -> None:
    arrow_root = tmp_path / "arrow"
    rows = [
        "ADM/train/ai/a.png",
        "ADM/val/ai/b.png",
        "stable_diffusion_v_1_4/train/nature/c.jpg",
        "stable_diffusion_v_1_4/val/nature/d.jpg",
        "Glide/train/ai/e.png",
        "Glide/val/ai/f.png",
        "BigGAN/train/ai/g.png",
        "BigGAN/val/ai/h.png",
        "Midjourney/train/ai/i.png",
        "Midjourney/val/ai/j.png",
        "stable_diffusion_v_1_5/train/ai/k.png",
        "wukong/val/ai/l.png",
        "VQDM/train/ai/m.png",
        "VQDM/val/ai/n.png",
    ]
    _write_arrow(arrow_root / "train/data-00000-of-00001.arrow", rows)
    index_root = tmp_path / "index"

    metadata = build_genimage_arrow_index(
        arrow_root, index_root, expected_shards=1
    )

    assert metadata["total_rows"] == len(rows)
    assert metadata["indexed_rows"] == len(rows)
    assert metadata["counts"]["train.SD"] == 1
    assert metadata["counts"]["val.SD"] == 1
    dataset = GenImageArrowClassDataset(
        arrow_root,
        index_root,
        "real",
        "train",
        expected_shards=1,
    )
    image, label = dataset[0]
    assert image.size == (4, 3)
    assert image.mode == "RGB"
    assert label == 0
