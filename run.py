from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from string import Formatter
from typing import Any, Iterable

from config import ExperimentConfig, load_config
from data import (
    EpisodePlan,
    build_episode_plan,
    load_manifest,
    scan_stage_folders,
    validate_files,
    validate_source_target_disjoint,
    write_manifest,
)
from methods.base import FewShotMethod
from train_fsd import train_fsd
from utils import BenchmarkError, ConfigurationError, binary_metrics, verify_backends


CORE_METHOD_NAMES = ("fsd", "ftnet", "ftnet_t")
EVALUATION_METHOD_NAMES = ("clipdet", "omnidfa_detection")
METHOD_NAMES = (*CORE_METHOD_NAMES, *EVALUATION_METHOD_NAMES)

METHOD_CONFIG_KEYS = {
    "fsd": {"checkpoint", "device", "batch_size", "fp16"},
    "ftnet": {
        "device",
        "backbone",
        "clip_layer",
        "download_root",
        "alpha",
        "batch_size",
    },
    "ftnet_t": {
        "device",
        "backbone",
        "clip_layer",
        "download_root",
        "alpha",
        "epochs",
        "learning_rate",
        "batch_size",
        "num_workers",
    },
    "clipdet": {"checkpoint", "backbone_checkpoint", "device", "batch_size"},
    "omnidfa_detection": {"checkpoint", "device", "batch_size", "dtype", "seed"},
}


def _validate_method_configs(config: ExperimentConfig) -> None:
    for name, method_config in config.methods.items():
        unknown = set(method_config) - METHOD_CONFIG_KEYS[name]
        if unknown:
            raise ConfigurationError(f"Unknown {name} settings: {sorted(unknown)}")

        if name in {"fsd", "clipdet", "omnidfa_detection"}:
            checkpoint = method_config.get("checkpoint")
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ConfigurationError(f"methods.{name}.checkpoint must be a non-empty path")
            try:
                fields = {
                    field_name
                    for _, field_name, _, _ in Formatter().parse(checkpoint)
                    if field_name is not None
                }
            except ValueError as exc:
                raise ConfigurationError(
                    f"methods.{name}.checkpoint contains invalid placeholder syntax"
                ) from exc
            allowed_fields = {"stage"} if name == "fsd" else set()
            if fields - allowed_fields:
                raise ConfigurationError(
                    f"methods.{name}.checkpoint contains unsupported placeholders"
                )

        if "device" in method_config:
            device = method_config["device"]
            if not isinstance(device, str) or not device.strip():
                raise ConfigurationError(f"methods.{name}.device must be a non-empty string")

        _validate_integer_setting(method_config, name, "batch_size", minimum=1)

        if (
            name == "fsd"
            and "fp16" in method_config
            and not isinstance(method_config["fp16"], bool)
        ):
            raise ConfigurationError("methods.fsd.fp16 must be true or false")

        if name in {"ftnet", "ftnet_t"}:
            _validate_integer_setting(method_config, name, "clip_layer", minimum=1)
            _validate_number_setting(method_config, name, "alpha", minimum_exclusive=0.0)
            if "backbone" in method_config:
                backbone = method_config["backbone"]
                if not isinstance(backbone, str) or not backbone.strip():
                    raise ConfigurationError(f"methods.{name}.backbone must be a non-empty string")
            if name == "ftnet_t":
                _validate_integer_setting(method_config, name, "num_workers", minimum=0)
                _validate_integer_setting(method_config, name, "epochs", minimum=1)
                _validate_number_setting(
                    method_config, name, "learning_rate", minimum_exclusive=0.0
                )
                epochs = method_config.get("epochs", 20)
                learning_rate = method_config.get("learning_rate", 0.001)
                if epochs != 20 or learning_rate != 0.001:
                    raise ConfigurationError(
                        "Exact FTNet-T requires epochs=20 and learning_rate=0.001"
                    )

        if name == "omnidfa_detection":
            _validate_integer_setting(method_config, name, "seed")
            dtype = method_config.get("dtype", "bfloat16")
            if dtype not in {"float32", "float16", "bfloat16"}:
                raise ConfigurationError(
                    "methods.omnidfa_detection.dtype must be float32, float16, or bfloat16"
                )

        for key in ("backbone_checkpoint", "download_root"):
            if key in method_config and (
                not isinstance(method_config[key], str) or not method_config[key].strip()
            ):
                raise ConfigurationError(f"methods.{name}.{key} must be a non-empty path")


def _validate_integer_setting(
    method_config: dict[str, Any], name: str, key: str, minimum: int | None = None
) -> None:
    if key not in method_config:
        return
    value = method_config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"methods.{name}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"methods.{name}.{key} must be at least {minimum}")


def _validate_number_setting(
    method_config: dict[str, Any], name: str, key: str, minimum_exclusive: float
) -> None:
    if key not in method_config:
        return
    value = method_config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"methods.{name}.{key} must be a finite number")
    if value <= minimum_exclusive:
        raise ConfigurationError(f"methods.{name}.{key} must be greater than {minimum_exclusive}")


def validate_experiment(config: ExperimentConfig) -> list[EpisodePlan]:
    unknown_configs = set(config.methods) - set(METHOD_NAMES)
    if unknown_configs:
        raise ConfigurationError(f"Unknown method configs: {sorted(unknown_configs)}")
    _validate_method_configs(config)
    if 0 in config.protocol.shots and set(config.methods) - set(EVALUATION_METHOD_NAMES):
        raise ConfigurationError("0-shot episodes are only valid for evaluation-only methods")
    samples = load_manifest(config.protocol.manifest)
    if config.protocol.source_generators:
        validate_source_target_disjoint(config.protocol.source_generators, config.protocol.stages)
    missing = validate_files(samples)
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise ConfigurationError(f"Manifest references {len(missing)} missing files:\n{preview}")
    return [
        build_episode_plan(
            samples,
            config.protocol.stages,
            shots,
            seed,
            config.protocol.query_per_class,
        )
        for shots in config.protocol.shots
        for seed in config.protocol.seeds
    ]


def run_experiment(
    config: ExperimentConfig,
    methods: Iterable[str] = METHOD_NAMES,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    verify_backends()
    plans = validate_experiment(config)
    selected = tuple(methods)
    if not selected:
        raise ConfigurationError("Select at least one method")
    if len(set(selected)) != len(selected):
        raise ConfigurationError("Method selection contains duplicates")
    unknown = set(selected) - set(METHOD_NAMES)
    if unknown:
        raise ConfigurationError(f"Unknown methods: {sorted(unknown)}")
    missing_method_configs = set(selected) - set(config.methods)
    if missing_method_configs:
        raise ConfigurationError(f"Missing method configs: {sorted(missing_method_configs)}")
    if any(plan.shots == 0 for plan in plans) and set(selected) - set(EVALUATION_METHOD_NAMES):
        raise ConfigurationError("0-shot episodes are only valid for evaluation-only methods")
    if (
        "fsd" in selected
        and config.protocol.evaluation_scope == "seen"
        and "{stage}" in str(config.methods["fsd"].get("checkpoint", ""))
    ):
        raise ConfigurationError(
            "A stage-specific FSD checkpoint cannot evaluate prior stages in one feature space; "
            "use evaluation_scope: current or one shared source checkpoint"
        )
    if (
        config.protocol.evaluation_scope == "seen"
        and not config.protocol.cumulative_cache
        and {"ftnet", "ftnet_t"} & set(selected)
    ):
        raise ConfigurationError("Seen-stage FTNet evaluation requires cumulative_cache: true")
    if config.protocol.episode_mode == "joint" and "fsd" in selected:
        raise ConfigurationError("FSD's released protocol is per-target, not a joint cache episode")
    if dry_run:
        return [_plan_summary(config, plan, selected) for plan in plans]

    all_records: list[dict[str, Any]] = []
    for method_name in selected:
        for plan in plans:
            method = _create_method(method_name, config)
            try:
                records = (
                    _run_joint_plan(config, plan, method)
                    if config.protocol.episode_mode == "joint"
                    else _run_plan(config, plan, method)
                )
                all_records.extend(records)
            finally:
                method.close()
    _write_results(config.output_dir, all_records)
    return all_records


def _create_method(name: str, config: ExperimentConfig) -> FewShotMethod:
    method_config = dict(config.methods[name])
    for key in ("checkpoint", "backbone_checkpoint"):
        if key in method_config:
            candidate = Path(str(method_config[key])).expanduser()
            if not candidate.is_absolute():
                method_config[key] = str(config.root / candidate)
    if name == "fsd":
        checkpoint = str(method_config["checkpoint"])
        method_config["checkpoint"] = checkpoint
        from methods.fsd import FSDMethod

        return FSDMethod(method_config)
    if name == "ftnet":
        from methods.ftnet import FTNetMethod

        return FTNetMethod(method_config)
    if name == "ftnet_t":
        from methods.ftnet import FTNetTMethod

        return FTNetTMethod(method_config)
    if name == "clipdet":
        from methods.clipdet import CLIPDetMethod

        return CLIPDetMethod(method_config)
    if name == "omnidfa_detection":
        from methods.omnidfa import OmniDFADetectionMethod

        return OmniDFADetectionMethod(method_config)
    raise AssertionError(name)


def _run_plan(
    config: ExperimentConfig, plan: EpisodePlan, method: FewShotMethod
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    run_root = config.output_dir / method.name / f"{plan.shots}shot" / f"seed_{plan.seed}"
    for stage in plan.stages:
        cumulative = (
            plan.support_through(stage.stage_index)
            if config.protocol.cumulative_cache
            else stage.support
        )
        artifact_dir = run_root / f"stage_{stage.stage_index:02d}_{_safe(stage.generator)}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / "episode.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generator": stage.generator,
                    "shots": plan.shots,
                    "seed": plan.seed,
                    "support": [
                        {"path": sample.identity, "label": sample.label} for sample in stage.support
                    ],
                    "query": [
                        {"path": sample.identity, "label": sample.label} for sample in stage.query
                    ],
                },
                handle,
                indent=2,
            )
        method.adapt(stage.generator, stage.support, cumulative, artifact_dir)

        stage_records = []
        evaluation_stages = (
            plan.stages[: stage.stage_index + 1]
            if config.protocol.evaluation_scope == "seen"
            else (stage,)
        )
        for evaluation_stage in evaluation_stages:
            probabilities = method.predict_fake_probability(
                evaluation_stage.generator, evaluation_stage.query
            )
            labels = [sample.label for sample in evaluation_stage.query]
            metrics = binary_metrics(labels, probabilities, method.decision_threshold)
            metrics.update(method.official_metrics(labels, probabilities))
            record: dict[str, Any] = {
                "experiment": config.name,
                "method": method.name,
                "adaptation_mode": method.adaptation_mode,
                "reproduction_scope": method.reproduction_scope,
                "decision_threshold": method.decision_threshold,
                "shots": plan.shots,
                "seed": plan.seed,
                "adapt_stage": stage.stage_index,
                "adapt_generator": stage.generator,
                "eval_stage": evaluation_stage.stage_index,
                "eval_generator": evaluation_stage.generator,
                **metrics,
            }
            records.append(record)
            stage_records.append(record)
        with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(stage_records, handle, indent=2, allow_nan=True)
    return records


def _run_joint_plan(
    config: ExperimentConfig, plan: EpisodePlan, method: FewShotMethod
) -> list[dict[str, Any]]:
    """Paper-style joint cache: K real/fake from every generator, then one evaluation."""
    run_root = config.output_dir / method.name / f"{plan.shots}shot" / f"seed_{plan.seed}"
    artifact_dir = run_root / "joint_episode"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    support = tuple(sample for stage in plan.stages for sample in stage.support)
    with (artifact_dir / "episode.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": "joint",
                "shots_per_class_per_generator": plan.shots,
                "seed": plan.seed,
                "support": [
                    {
                        "path": sample.identity,
                        "label": sample.label,
                        "generator": sample.generator,
                    }
                    for sample in support
                ],
                "query": {
                    stage.generator: [
                        {"path": sample.identity, "label": sample.label} for sample in stage.query
                    ]
                    for stage in plan.stages
                },
            },
            handle,
            indent=2,
        )
    method.adapt("joint", support, support, artifact_dir)

    records = []
    for stage in plan.stages:
        scores = method.predict_fake_probability(stage.generator, stage.query)
        metrics = binary_metrics(
            [sample.label for sample in stage.query], scores, method.decision_threshold
        )
        metrics.update(method.official_metrics([sample.label for sample in stage.query], scores))
        records.append(
            {
                "experiment": config.name,
                "method": method.name,
                "adaptation_mode": method.adaptation_mode,
                "reproduction_scope": method.reproduction_scope,
                "decision_threshold": method.decision_threshold,
                "shots": plan.shots,
                "seed": plan.seed,
                "adapt_stage": -1,
                "adapt_generator": "joint",
                "eval_stage": stage.stage_index,
                "eval_generator": stage.generator,
                **metrics,
            }
        )
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, allow_nan=True)
    return records


def _write_results(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=True) + "\n")
    if records:
        with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(dict.fromkeys(key for record in records for key in record))
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        _write_summary(output_dir / "summary.csv", records)


def _write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["method"], record["shots"], record["seed"], record["adapt_stage"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (method, shots, seed, stage), values in sorted(groups.items()):
        rows.append(
            {
                "method": method,
                "shots": shots,
                "seed": seed,
                "adapt_stage": stage,
                "seen_mean_accuracy": sum(item["accuracy"] for item in values) / len(values),
                "seen_mean_balanced_accuracy": sum(
                    item.get("balanced_accuracy", item["accuracy"]) for item in values
                )
                / len(values),
                "seen_mean_average_precision": sum(item["average_precision"] for item in values)
                / len(values),
                "seen_mean_roc_auc": sum(item["roc_auc"] for item in values) / len(values),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plan_summary(
    config: ExperimentConfig, plan: EpisodePlan, methods: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "experiment": config.name,
        "methods": list(methods),
        "shots": plan.shots,
        "seed": plan.seed,
        "episode_mode": config.protocol.episode_mode,
        "stages": [
            {
                "index": stage.stage_index,
                "generator": stage.generator,
                "support": len(stage.support),
                "query": len(stage.query),
            }
            for stage in plan.stages
        ],
    }


def _safe(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aigcd", description="Few-shot AIGC benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Verify official repository commits and file hashes")

    index = subparsers.add_parser("index", help="Create a manifest from stage folders")
    index.add_argument("--data-root", required=True, type=Path)
    index.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser("validate", help="Validate config, data, and all splits")
    validate.add_argument("--config", required=True, type=Path)

    run = subparsers.add_parser("run", help="Run one or all comparison methods")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--method", choices=("all", *METHOD_NAMES), default="all")
    run.add_argument("--dry-run", action="store_true")

    train = subparsers.add_parser("train-fsd", help="Train the integrated FSD implementation")
    train.add_argument("--data-root", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--exclude-class", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--total-steps", type=int, default=200_000)
    train.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            print(json.dumps(verify_backends(strict=False), indent=2))
            if not all(item["ok"] for item in verify_backends(strict=False)):
                raise SystemExit(1)
        elif args.command == "index":
            samples = scan_stage_folders(args.data_root)
            write_manifest(samples, args.output, relative_to=args.output.parent)
            print(f"Wrote {len(samples)} rows to {args.output.resolve()}")
        elif args.command == "validate":
            config = load_config(args.config)
            plans = validate_experiment(config)
            print(json.dumps({"valid": True, "plans": len(plans)}, indent=2))
        elif args.command == "run":
            config = load_config(args.config)
            methods = tuple(config.methods) if args.method == "all" else (args.method,)
            records = run_experiment(config, methods, dry_run=args.dry_run)
            print(json.dumps(records, indent=2, allow_nan=True))
        elif args.command == "train-fsd":
            checkpoint = train_fsd(
                args.data_root,
                args.output_dir,
                args.exclude_class,
                args.device,
                args.workers,
                args.seed,
                args.total_steps,
                args.batch_size,
            )
            print(checkpoint)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
