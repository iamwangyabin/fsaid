from __future__ import annotations

import json
import re
from collections import OrderedDict
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any


GENIMAGE_ARROW_INDEX_VERSION = 3
GENIMAGE_ARROW_SHARDS = 1214
GENIMAGE_FSD_CLASSES = ("real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")

_SHARD_PATTERN = re.compile(r"data-(\d{5})-of-(\d{5})\.arrow$")
_FAKE_SOURCE_CLASSES = {
    "ADM": "ADM",
    "BigGAN": "BigGAN",
    "Glide": "glide",
    "glide": "glide",
    "Midjourney": "Midjourney",
    "VQDM": "VQDM",
    "stable_diffusion_v_1_4": "SD",
    "stable_diffusion_v_1_5": "SD",
    "wukong": "SD",
}
_REAL_SOURCES = {"stable_diffusion_v_1_4", "stable_diffusion_v_1_5"}


def classify_genimage_arrow_path(image_path: str) -> tuple[str, str] | None:
    """Index a raw GenImage path without changing its recorded split."""
    parts = PurePosixPath(image_path).parts
    if len(parts) < 3:
        return None
    source, split, category = parts[:3]
    if split not in {"train", "val"}:
        return None
    if category == "ai" and source in _FAKE_SOURCE_CLASSES:
        return split, _FAKE_SOURCE_CLASSES[source]
    if category == "nature" and source in _REAL_SOURCES:
        return split, "real"
    return None


def _arrow_shards(arrow_root: Path, expected_shards: int | None) -> list[Path]:
    train_root = arrow_root.resolve() / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"GenImage Arrow train directory is missing: {train_root}")
    shards = sorted(train_root.glob("data-*-of-*.arrow"))
    if not shards:
        raise FileNotFoundError(f"No GenImage Arrow shards found in {train_root}")

    parsed: list[tuple[int, int]] = []
    for shard in shards:
        match = _SHARD_PATTERN.fullmatch(shard.name)
        if match is None:
            raise ValueError(f"Unexpected GenImage Arrow shard name: {shard.name}")
        parsed.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _, total in parsed}
    if len(totals) != 1:
        raise ValueError(f"GenImage Arrow shards disagree on total count: {sorted(totals)}")
    declared_total = totals.pop()
    required_total = declared_total if expected_shards is None else expected_shards
    if declared_total != required_total:
        raise ValueError(
            f"GenImage Arrow declares {declared_total} shards, expected {required_total}"
        )
    indices = [index for index, _ in parsed]
    if indices != list(range(required_total)):
        missing = sorted(set(range(required_total)) - set(indices))
        raise ValueError(
            f"GenImage Arrow view is incomplete: found {len(indices)}/{required_total} shards; "
            f"first missing indices: {missing[:5]}"
        )
    return shards


def build_genimage_arrow_index(
    arrow_root: Path,
    index_root: Path,
    expected_shards: int | None = GENIMAGE_ARROW_SHARDS,
    require_all_classes: bool = True,
) -> dict[str, Any]:
    """Build compact row locators without reading or duplicating image payloads."""
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("GenImage Arrow indexing requires numpy and pyarrow") from exc

    shards = _arrow_shards(arrow_root, expected_shards)
    locators: dict[tuple[str, str], list[int]] = {
        (split, class_name): []
        for split in ("train", "val")
        for class_name in GENIMAGE_FSD_CLASSES
    }
    ignored_rows = 0
    unknown_fake_sources: set[str] = set()
    total_rows = 0

    for shard_position, shard in enumerate(shards):
        source = pa.memory_map(str(shard), "r")
        try:
            reader = ipc.open_stream(source)
            path_column = reader.schema.get_field_index("image_path")
            if path_column < 0 or reader.schema.get_field_index("image") < 0:
                raise ValueError(f"GenImage Arrow shard lacks image_path/image columns: {shard}")
            row_offset = 0
            for batch in reader:
                paths = batch.column(path_column).to_pylist()
                for batch_offset, image_path in enumerate(paths):
                    mapped = classify_genimage_arrow_path(image_path)
                    if mapped is None:
                        parts = PurePosixPath(image_path).parts
                        if len(parts) >= 3 and parts[2] == "ai":
                            unknown_fake_sources.add(parts[0])
                        ignored_rows += 1
                        continue
                    row = row_offset + batch_offset
                    locator = (shard_position << 32) | row
                    locators[mapped].append(locator)
                row_offset += batch.num_rows
            total_rows += row_offset
        finally:
            source.close()

    if unknown_fake_sources:
        raise ValueError(
            "Unmapped GenImage fake sources: " + ", ".join(sorted(unknown_fake_sources))
        )
    empty = [
        f"train/{name}"
        for name in GENIMAGE_FSD_CLASSES
        if not locators[("train", name)]
    ]
    if empty and require_all_classes:
        raise ValueError(f"GenImage Arrow index has empty logical classes: {empty}")

    index_root = index_root.resolve()
    index_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for (split, class_name), rows in locators.items():
        key = f"{split}.{class_name}"
        destination = index_root / f"{key}.npy"
        temporary = destination.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(rows, dtype=np.uint64), allow_pickle=False)
        temporary.replace(destination)
        counts[key] = len(rows)

    metadata = {
        "format_version": GENIMAGE_ARROW_INDEX_VERSION,
        "arrow_root": str(arrow_root.resolve()),
        "shards": [
            {"name": shard.name, "size": shard.stat().st_size}
            for shard in shards
        ],
        "total_rows": total_rows,
        "indexed_rows": sum(counts.values()),
        "ignored_rows": ignored_rows,
        "counts": counts,
    }
    metadata_path = index_root / "metadata.json"
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return metadata


def load_genimage_arrow_index(
    arrow_root: Path,
    index_root: Path,
    expected_shards: int | None = GENIMAGE_ARROW_SHARDS,
) -> dict[str, Any]:
    metadata_path = index_root.resolve() / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"GenImage Arrow index metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != GENIMAGE_ARROW_INDEX_VERSION:
        raise ValueError(f"Unsupported GenImage Arrow index format: {metadata.get('format_version')}")

    shards = _arrow_shards(arrow_root, expected_shards)
    recorded = metadata.get("shards")
    current = [{"name": shard.name, "size": shard.stat().st_size} for shard in shards]
    if recorded != current:
        raise ValueError("GenImage Arrow index does not match the current shard names and sizes")
    for split in ("train", "val"):
        for class_name in GENIMAGE_FSD_CLASSES:
            path = index_root.resolve() / f"{split}.{class_name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"GenImage Arrow locator file is missing: {path}")
    return metadata


def ensure_genimage_arrow_index(
    arrow_root: Path,
    index_root: Path,
    expected_shards: int | None = GENIMAGE_ARROW_SHARDS,
) -> dict[str, Any]:
    try:
        return load_genimage_arrow_index(arrow_root, index_root, expected_shards)
    except FileNotFoundError:
        return build_genimage_arrow_index(arrow_root, index_root, expected_shards)


class GenImageArrowClassDataset:
    """Random-access logical FSD class backed by memory-mapped Arrow shards."""

    def __init__(
        self,
        arrow_root: Path,
        index_root: Path,
        class_name: str,
        split: str,
        transform: Any = None,
        max_open_shards: int = 64,
        expected_shards: int | None = GENIMAGE_ARROW_SHARDS,
    ) -> None:
        if class_name not in GENIMAGE_FSD_CLASSES:
            raise ValueError(f"Unknown GenImage FSD class: {class_name}")
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported GenImage split: {split}")
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")

        metadata = load_genimage_arrow_index(arrow_root, index_root, expected_shards)
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("GenImage Arrow loading requires numpy") from exc

        self.arrow_root = arrow_root.resolve()
        self.index_root = index_root.resolve()
        self.class_name = class_name
        self.split = split
        self.transform = transform
        self.max_open_shards = max_open_shards
        self.shards = [self.arrow_root / "train" / item["name"] for item in metadata["shards"]]
        self.locators = np.load(
            self.index_root / f"{split}.{class_name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self._cache: OrderedDict[int, tuple[Any, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.locators)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _open_shard(self, shard_position: int) -> Any:
        cached = self._cache.pop(shard_position, None)
        if cached is not None:
            self._cache[shard_position] = cached
            return cached[1]
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise RuntimeError("GenImage Arrow loading requires pyarrow") from exc

        source = pa.memory_map(str(self.shards[shard_position]), "r")
        table = ipc.open_stream(source).read_all().select(["image"])
        self._cache[shard_position] = (source, table)
        if len(self._cache) > self.max_open_shards:
            _, (old_source, old_table) = self._cache.popitem(last=False)
            del old_table
            old_source.close()
        return table

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        locator = int(self.locators[index])
        shard_position = locator >> 32
        row = locator & 0xFFFFFFFF
        table = self._open_shard(shard_position)
        payload = table.column("image")[row].as_py()
        if not isinstance(payload, bytes):
            raise ValueError(
                f"GenImage Arrow image payload is not bytes: shard={shard_position}, row={row}"
            )

        from PIL import Image

        with Image.open(BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
