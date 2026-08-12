import csv
from pathlib import Path

from data import Sample, load_manifest, write_manifest
from scripts.prepare_cross_benchmark_manifest import prepare_manifest


def test_reserves_fixed_support_candidates_and_query(tmp_path: Path) -> None:
    samples = [
        Sample(tmp_path / f"{generator}-{label}-{index}.png", label, generator)
        for generator in ("a", "b")
        for label in (0, 1)
        for index in range(8)
    ]
    for sample in samples:
        sample.path.touch()
    source = tmp_path / "source.csv"
    output = tmp_path / "cross.csv"
    write_manifest(samples, source)

    summary = prepare_manifest(source, output, support_candidates_per_class=3, reserve_seed=7)
    prepared = load_manifest(output)

    assert summary["rows"] == 32
    for generator in ("a", "b"):
        for label in (0, 1):
            group = [
                sample
                for sample in prepared
                if sample.generator == generator and sample.label == label
            ]
            assert sum(sample.split == "support" for sample in group) == 3
            assert sum(sample.split == "query" for sample in group) == 5


def test_output_is_deterministic(tmp_path: Path) -> None:
    samples = [
        Sample(tmp_path / f"a-{label}-{index}.png", label, "a")
        for label in (0, 1)
        for index in range(6)
    ]
    source = tmp_path / "source.csv"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_manifest(samples, source)

    prepare_manifest(source, first, support_candidates_per_class=2, reserve_seed=9)
    prepare_manifest(source, second, support_candidates_per_class=2, reserve_seed=9)

    with first.open(newline="", encoding="utf-8") as first_handle, second.open(
        newline="", encoding="utf-8"
    ) as second_handle:
        assert list(csv.DictReader(first_handle)) == list(csv.DictReader(second_handle))
