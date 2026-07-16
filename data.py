from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from utils import ConfigurationError, DataLeakageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_COLUMNS = ("path", "label", "generator", "split")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    generator: str
    split: str = "pool"
    sample_id: str | None = None

    @property
    def identity(self) -> str:
        return str(self.path.resolve())

    @property
    def split_key(self) -> str:
        return self.sample_id or self.identity


def load_manifest(path: str | Path) -> list[Sample]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ConfigurationError(f"Manifest does not exist: {manifest_path}")

    samples: list[Sample] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or ()
        missing = set(MANIFEST_COLUMNS) - set(fieldnames)
        if missing:
            raise ConfigurationError(f"Manifest is missing columns: {sorted(missing)}")
        if len(set(fieldnames)) != len(fieldnames):
            raise ConfigurationError("Manifest contains duplicate column names")
        for line_number, row in enumerate(reader, start=2):
            path_value = row.get("path")
            if not isinstance(path_value, str) or not path_value.strip():
                raise ConfigurationError(f"Empty path at line {line_number}")
            path_value = path_value.strip()
            raw_path = Path(path_value).expanduser()
            image_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
            try:
                label = int(row.get("label"))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"Invalid label at line {line_number}") from exc
            if label not in (0, 1):
                raise ConfigurationError(f"Label must be 0 or 1 at line {line_number}")
            generator_value = row.get("generator")
            if not isinstance(generator_value, str):
                raise ConfigurationError(f"Empty generator at line {line_number}")
            generator = generator_value.strip()
            if not generator:
                raise ConfigurationError(f"Empty generator at line {line_number}")
            split_value = row.get("split")
            if not isinstance(split_value, str):
                raise ConfigurationError(f"Missing split at line {line_number}")
            split = split_value.strip().lower() or "pool"
            if split not in {"pool", "support", "query"}:
                raise ConfigurationError(
                    f"split must be pool/support/query at line {line_number}, got {split!r}"
                )
            samples.append(
                Sample(
                    image_path.resolve(),
                    label,
                    generator,
                    split,
                    path_value.replace("\\", "/"),
                )
            )

    duplicate_rows = _duplicates(sample.identity for sample in samples)
    if duplicate_rows:
        raise DataLeakageError(f"Manifest repeats image paths: {duplicate_rows[:5]}")
    return samples


def write_manifest(
    samples: Sequence[Sample], path: str | Path, relative_to: Path | None = None
) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for sample in samples:
            value = sample.path
            if relative_to is not None:
                try:
                    value = sample.path.relative_to(relative_to.resolve())
                except ValueError:
                    pass
            writer.writerow(
                {
                    "path": str(value),
                    "label": sample.label,
                    "generator": sample.generator,
                    "split": sample.split,
                }
            )


def scan_stage_folders(root: str | Path) -> list[Sample]:
    """Scan ROOT/<generator>/{real|0_real,fake|1_fake}/**/* into a manifest."""
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ConfigurationError(f"Dataset root does not exist: {dataset_root}")

    samples: list[Sample] = []
    aliases = {"real": 0, "0_real": 0, "fake": 1, "1_fake": 1}
    for generator_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        found_labels: set[int] = set()
        for class_dir in sorted(path for path in generator_dir.iterdir() if path.is_dir()):
            label = aliases.get(class_dir.name.lower())
            if label is None:
                continue
            found_labels.add(label)
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(Sample(image_path.resolve(), label, generator_dir.name, "pool"))
        if found_labels != {0, 1}:
            raise ConfigurationError(
                f"{generator_dir} must contain both real/0_real and fake/1_fake directories"
            )
    if not samples:
        raise ConfigurationError(f"No images found under {dataset_root}")
    return samples


def validate_files(samples: Iterable[Sample]) -> list[Path]:
    return [sample.path for sample in samples if not sample.path.is_file()]


def stable_order(samples: Sequence[Sample], seed: int, namespace: str) -> list[Sample]:
    def key(sample: Sample) -> bytes:
        payload = f"{seed}\0{namespace}\0{sample.split_key}".encode()
        return hashlib.sha256(payload).digest()

    return sorted(samples, key=key)


@dataclass(frozen=True)
class StageEpisode:
    stage_index: int
    generator: str
    support: tuple[Sample, ...]
    query: tuple[Sample, ...]


@dataclass(frozen=True)
class EpisodePlan:
    shots: int
    seed: int
    stages: tuple[StageEpisode, ...]

    def support_through(self, stage_index: int) -> tuple[Sample, ...]:
        return tuple(sample for stage in self.stages[: stage_index + 1] for sample in stage.support)


def build_episode_plan(
    samples: Sequence[Sample],
    stage_order: Sequence[str],
    shots: int,
    seed: int,
    query_per_class: int | None = None,
) -> EpisodePlan:
    if shots < 0:
        raise ConfigurationError("shots must be non-negative")
    if query_per_class is not None and query_per_class <= 0:
        raise ConfigurationError("query_per_class must be positive or None")
    episodes: list[StageEpisode] = []
    for stage_index, generator in enumerate(stage_order):
        stage_samples = [sample for sample in samples if sample.generator == generator]
        if not stage_samples:
            raise ConfigurationError(f"No manifest rows found for stage {generator!r}")

        support: list[Sample] = []
        query: list[Sample] = []
        for label in (0, 1):
            label_samples = [sample for sample in stage_samples if sample.label == label]
            explicit_support = [sample for sample in label_samples if sample.split == "support"]
            explicit_query = [sample for sample in label_samples if sample.split == "query"]
            pool = [sample for sample in label_samples if sample.split == "pool"]

            if explicit_support or explicit_query:
                if pool:
                    raise ConfigurationError(
                        f"Stage {generator}, label {label}: do not mix explicit splits with pool"
                    )
                support_candidates = stable_order(
                    explicit_support, seed, f"{generator}:{label}:support"
                )
                if len(support_candidates) < shots:
                    raise ConfigurationError(
                        f"Stage {generator}, label {label}: need {shots} support images, "
                        f"found {len(support_candidates)}"
                    )
                chosen_support = support_candidates[:shots]
                query_candidates = stable_order(explicit_query, seed, f"{generator}:{label}:query")
            else:
                ordered = stable_order(pool, seed, f"{generator}:{label}:pool")
                if len(ordered) <= shots:
                    raise ConfigurationError(
                        f"Stage {generator}, label {label}: need more than {shots} pooled images"
                    )
                chosen_support = ordered[:shots]
                query_candidates = ordered[shots:]

            if not query_candidates:
                raise ConfigurationError(f"Stage {generator}, label {label}: query split is empty")
            if query_per_class is not None:
                if len(query_candidates) < query_per_class:
                    raise ConfigurationError(
                        f"Stage {generator}, label {label}: need {query_per_class} query images, "
                        f"found {len(query_candidates)}"
                    )
                query_candidates = query_candidates[:query_per_class]
            support.extend(chosen_support)
            query.extend(query_candidates)

        _assert_disjoint(support, query, generator)
        episodes.append(StageEpisode(stage_index, generator, tuple(support), tuple(query)))

    return EpisodePlan(shots, seed, tuple(episodes))


def validate_source_target_disjoint(
    source_generators: Iterable[str], stages: Sequence[str]
) -> None:
    overlap = set(source_generators) & set(stages)
    if overlap:
        raise DataLeakageError(f"Source and incremental generators overlap: {sorted(overlap)}")


def _assert_disjoint(support: Sequence[Sample], query: Sequence[Sample], generator: str) -> None:
    overlap = {sample.identity for sample in support} & {sample.identity for sample in query}
    if overlap:
        raise DataLeakageError(
            f"Stage {generator} has support/query leakage: {sorted(overlap)[:5]}"
        )


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    return duplicates
