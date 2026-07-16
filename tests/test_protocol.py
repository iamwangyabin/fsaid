from pathlib import Path

import pytest

from data import Sample, build_episode_plan, validate_source_target_disjoint
from utils import ConfigurationError, DataLeakageError


def samples(tmp_path: Path) -> list[Sample]:
    return [
        Sample(tmp_path / f"stage-a-{label}-{index}.png", label, "stage-a")
        for label in (0, 1)
        for index in range(8)
    ]


def test_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    first = build_episode_plan(samples(tmp_path), ["stage-a"], shots=2, seed=7)
    second = build_episode_plan(list(reversed(samples(tmp_path))), ["stage-a"], shots=2, seed=7)
    assert first == second
    support = {sample.identity for sample in first.stages[0].support}
    query = {sample.identity for sample in first.stages[0].query}
    assert support.isdisjoint(query)
    assert len(support) == 4
    assert len(query) == 12


def test_explicit_split_cannot_mix_with_pool(tmp_path: Path) -> None:
    rows = samples(tmp_path)
    rows[0] = Sample(rows[0].path, rows[0].label, rows[0].generator, "support")
    with pytest.raises(ConfigurationError, match="do not mix"):
        build_episode_plan(rows, ["stage-a"], shots=1, seed=0)


def test_source_target_leakage_is_rejected() -> None:
    with pytest.raises(DataLeakageError, match="overlap"):
        validate_source_target_disjoint(["ADM", "SD1.5"], ["SD1.5", "SDXL"])


def test_zero_shot_keeps_the_full_pool_as_query(tmp_path: Path) -> None:
    plan = build_episode_plan(samples(tmp_path), ["stage-a"], shots=0, seed=0)
    assert plan.stages[0].support == ()
    assert len(plan.stages[0].query) == 16


def test_explicit_support_without_query_is_rejected(tmp_path: Path) -> None:
    rows = [Sample(row.path, row.label, row.generator, "support") for row in samples(tmp_path)]
    with pytest.raises(ConfigurationError, match="query split is empty"):
        build_episode_plan(rows, ["stage-a"], shots=1, seed=0)
