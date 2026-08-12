import json
from pathlib import Path

import pytest
from PIL import Image

from config import ExperimentConfig, ProtocolConfig
from data import Sample, build_episode_plan, write_manifest
from methods.base import FewShotMethod
from run import episode_path, run_experiment, run_joint_plan, write_episode_plans, write_results
from utils import ConfigurationError


class _RecordingMethod(FewShotMethod):
    name = "recording"

    def __init__(self) -> None:
        self.support: tuple[Sample, ...] = ()

    def adapt(self, generator, stage_support, cumulative_support, artifact_dir) -> None:
        assert generator == "joint"
        assert tuple(stage_support) == tuple(cumulative_support)
        self.support = tuple(stage_support)

    def predict_fake_probability(self, generator, samples) -> list[float]:
        return [float(sample.label) for sample in samples]


def test_dry_run_validates_full_shared_plan(tmp_path: Path) -> None:
    samples = []
    for generator in ("g1", "g2"):
        for label in (0, 1):
            for index in range(3):
                path = tmp_path / f"{generator}-{label}-{index}.png"
                Image.new("RGB", (8, 8)).save(path)
                samples.append(Sample(path, label, generator))
    manifest = tmp_path / "manifest.csv"
    write_manifest(samples, manifest)
    protocol = ProtocolConfig(
        manifest=manifest,
        stages=("g1", "g2"),
        shots=(1,),
        seeds=(5,),
        query_per_class=None,
        cumulative_cache=True,
        evaluation_scope="seen",
        source_generators=(),
    )
    config = ExperimentConfig(
        name="test",
        root=tmp_path,
        output_dir=tmp_path / "outputs",
        protocol=protocol,
        methods={"fsd": {"checkpoint": "unused.pth"}},
    )
    result = run_experiment(config, methods=("fsd",), dry_run=True)
    assert result[0]["stages"] == [
        {"index": 0, "generator": "g1", "support": 2, "query": 4},
        {"index": 1, "generator": "g2", "support": 2, "query": 4},
    ]


def test_joint_mode_is_reported_in_dry_run(tmp_path: Path) -> None:
    samples = []
    for generator in ("g1", "g2"):
        for label in (0, 1):
            for index in range(2):
                path = tmp_path / f"{generator}-{label}-{index}.png"
                Image.new("RGB", (8, 8)).save(path)
                samples.append(Sample(path, label, generator))
    manifest = tmp_path / "manifest.csv"
    write_manifest(samples, manifest)
    config = ExperimentConfig(
        name="joint",
        root=tmp_path,
        output_dir=tmp_path / "outputs",
        protocol=ProtocolConfig(
            manifest=manifest,
            stages=("g1", "g2"),
            shots=(1,),
            seeds=(40,),
            query_per_class=None,
            cumulative_cache=True,
            evaluation_scope="current",
            source_generators=(),
            episode_mode="joint",
        ),
        methods={"ftnet": {}},
    )
    result = run_experiment(config, methods=("ftnet",), dry_run=True)
    assert result[0]["episode_mode"] == "joint"


def test_joint_mode_adapts_once_with_all_generator_support(tmp_path: Path) -> None:
    samples = []
    for generator in ("g1", "g2"):
        for label in (0, 1):
            for index in range(3):
                path = tmp_path / f"{generator}-{label}-{index}.png"
                Image.new("RGB", (8, 8)).save(path)
                samples.append(Sample(path, label, generator))
    plan = build_episode_plan(samples, ("g1", "g2"), shots=1, seed=40)
    config = ExperimentConfig(
        name="joint",
        root=tmp_path,
        output_dir=tmp_path / "outputs",
        protocol=ProtocolConfig(
            manifest=tmp_path / "unused.csv",
            stages=("g1", "g2"),
            shots=(1,),
            seeds=(40,),
            query_per_class=None,
            cumulative_cache=True,
            evaluation_scope="current",
            source_generators=(),
            episode_mode="joint",
        ),
        methods={"ftnet": {}},
    )
    method = _RecordingMethod()

    records = run_joint_plan(config, plan, method)

    assert len(method.support) == 4
    assert {sample.generator for sample in method.support} == {"g1", "g2"}
    assert [record["eval_generator"] for record in records] == ["g1", "g2"]
    assert all(record["accuracy"] == 1.0 for record in records)
    assert all(record["reproduction_scope"] == method.reproduction_scope for record in records)


def test_episode_plan_is_stored_once_outside_method_directories(tmp_path: Path) -> None:
    samples = [
        Sample(tmp_path / f"{label}-{index}.png", label, "g1")
        for label in (0, 1)
        for index in range(2)
    ]
    plan = build_episode_plan(samples, ("g1",), shots=1, seed=5)
    config = ExperimentConfig(
        name="episodes",
        root=tmp_path,
        output_dir=tmp_path / "outputs",
        protocol=ProtocolConfig(
            manifest=tmp_path / "unused.csv",
            stages=("g1",),
            shots=(1,),
            seeds=(5,),
            query_per_class=None,
            cumulative_cache=False,
            evaluation_scope="current",
            source_generators=(),
        ),
        methods={"ftnet": {}, "ftnet_t": {}},
    )

    write_episode_plans(config, (plan,))

    path = episode_path(config, plan)
    assert path == tmp_path / "outputs/episodes/continual/1shot/seed_5/episode.json"
    assert json.loads(path.read_text(encoding="utf-8"))["stages"][0]["generator"] == "g1"
    assert not list((tmp_path / "outputs").glob("ftnet*/**/episode.json"))


def test_result_csv_accepts_method_specific_metric_columns(tmp_path: Path) -> None:
    records = [
        {
            "method": "a",
            "shots": 0,
            "seed": 0,
            "adapt_stage": -1,
            "accuracy": 1.0,
            "average_precision": 1.0,
            "roc_auc": 1.0,
        },
        {
            "method": "b",
            "shots": 0,
            "seed": 0,
            "adapt_stage": -1,
            "accuracy": 1.0,
            "average_precision": 1.0,
            "roc_auc": 1.0,
            "official_metric": 0.9,
        },
    ]
    write_results(tmp_path, records)
    header = (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "official_metric" in header


def test_zero_shot_is_rejected_for_few_shot_methods(tmp_path: Path) -> None:
    samples = []
    for label in (0, 1):
        path = tmp_path / f"g1-{label}.png"
        Image.new("RGB", (8, 8)).save(path)
        samples.append(Sample(path, label, "g1"))
    manifest = tmp_path / "manifest.csv"
    write_manifest(samples, manifest)
    config = ExperimentConfig(
        name="invalid-zero-shot",
        root=tmp_path,
        output_dir=tmp_path / "outputs",
        protocol=ProtocolConfig(
            manifest=manifest,
            stages=("g1",),
            shots=(0,),
            seeds=(0,),
            query_per_class=None,
            cumulative_cache=False,
            evaluation_scope="current",
            source_generators=(),
        ),
        methods={"fsd": {"checkpoint": "unused.pth"}},
    )
    with pytest.raises(ConfigurationError, match="evaluation-only"):
        run_experiment(config, methods=("fsd",), dry_run=True)


def test_unknown_method_setting_is_rejected_before_execution(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="invalid-setting",
        root=tmp_path,
        output_dir=tmp_path / "outputs",
        protocol=ProtocolConfig(
            manifest=tmp_path / "unused.csv",
            stages=("g1",),
            shots=(1,),
            seeds=(0,),
            query_per_class=None,
            cumulative_cache=False,
            evaluation_scope="current",
            source_generators=(),
        ),
        methods={"ftnet": {"batch_szie": 32}},
    )
    with pytest.raises(ConfigurationError, match="Unknown ftnet settings"):
        run_experiment(config, methods=("ftnet",), dry_run=True)
