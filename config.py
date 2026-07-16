from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from utils import ConfigurationError


@dataclass(frozen=True)
class ProtocolConfig:
    manifest: Path
    stages: tuple[str, ...]
    shots: tuple[int, ...]
    seeds: tuple[int, ...]
    query_per_class: int | None
    cumulative_cache: bool
    evaluation_scope: str
    source_generators: tuple[str, ...]
    episode_mode: str = "continual"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    root: Path
    output_dir: Path
    protocol: ProtocolConfig
    methods: dict[str, dict[str, Any]]


def _required(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing '{key}' in {where}")
    return mapping[key]


def _yaml_list(mapping: dict[str, Any], key: str, where: str) -> list[Any]:
    value = _required(mapping, key, where)
    if not isinstance(value, list):
        raise ConfigurationError(f"{where}.{key} must be a YAML list")
    return value


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{where} must be an integer")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{where} must be true or false")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ConfigurationError(f"Cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Root config must be a YAML mapping")

    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    protocol_raw = _required(raw, "protocol", "root config")
    if not isinstance(protocol_raw, dict):
        raise ConfigurationError("protocol must be a YAML mapping")
    stage_values = _yaml_list(protocol_raw, "stages", "protocol")
    if any(not isinstance(item, str) for item in stage_values):
        raise ConfigurationError("protocol.stages must contain only strings")
    stages = tuple(item.strip() for item in stage_values)
    shots = tuple(
        _integer(item, "Each protocol.shots value")
        for item in _yaml_list(protocol_raw, "shots", "protocol")
    )
    seeds = tuple(
        _integer(item, "Each protocol.seeds value")
        for item in _yaml_list(protocol_raw, "seeds", "protocol")
    )
    if not stages or not shots or not seeds:
        raise ConfigurationError("protocol.stages, shots, and seeds must be non-empty")
    if any(shot < 0 for shot in shots):
        raise ConfigurationError("All shot counts must be non-negative")
    if any(not stage.strip() for stage in stages):
        raise ConfigurationError("protocol.stages cannot contain empty names")
    if len(set(stages)) != len(stages):
        raise ConfigurationError("protocol.stages contains duplicates")
    if len(set(shots)) != len(shots):
        raise ConfigurationError("protocol.shots contains duplicates")
    if len(set(seeds)) != len(seeds):
        raise ConfigurationError("protocol.seeds contains duplicates")

    def resolve(value: Any, where: str) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ConfigurationError(f"{where} must be a non-empty path")
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    evaluation_scope = str(protocol_raw.get("evaluation_scope", "seen"))
    if evaluation_scope not in {"seen", "current"}:
        raise ConfigurationError("protocol.evaluation_scope must be 'seen' or 'current'")
    episode_mode = str(protocol_raw.get("episode_mode", "continual"))
    if episode_mode not in {"continual", "joint"}:
        raise ConfigurationError("protocol.episode_mode must be 'continual' or 'joint'")
    query_value = protocol_raw.get("query_per_class")
    query_per_class = (
        None if query_value is None else _integer(query_value, "protocol.query_per_class")
    )
    if query_per_class is not None and query_per_class <= 0:
        raise ConfigurationError("protocol.query_per_class must be positive or null")
    source_values = protocol_raw.get("source_generators", [])
    if not isinstance(source_values, list) or any(
        not isinstance(item, str) for item in source_values
    ):
        raise ConfigurationError("protocol.source_generators must be a YAML list of strings")
    source_generators = tuple(item.strip() for item in source_values)
    if any(not generator for generator in source_generators):
        raise ConfigurationError("protocol.source_generators cannot contain empty names")
    if len(set(source_generators)) != len(source_generators):
        raise ConfigurationError("protocol.source_generators contains duplicates")

    methods = _required(raw, "methods", "root config")
    if not isinstance(methods, dict) or not methods:
        raise ConfigurationError("methods must be a non-empty YAML mapping")
    if any(not isinstance(value, dict) for value in methods.values()):
        raise ConfigurationError("Every method config must be a YAML mapping")
    if any(not isinstance(key, str) or not key.strip() for key in methods):
        raise ConfigurationError("Method names must be non-empty strings")

    name = raw.get("name", config_path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("name must be a non-empty string")

    protocol = ProtocolConfig(
        manifest=resolve(_required(protocol_raw, "manifest", "protocol"), "protocol.manifest"),
        stages=stages,
        shots=shots,
        seeds=seeds,
        query_per_class=query_per_class,
        cumulative_cache=_boolean(
            protocol_raw.get("cumulative_cache", True), "protocol.cumulative_cache"
        ),
        evaluation_scope=evaluation_scope,
        source_generators=source_generators,
        episode_mode=episode_mode,
    )
    return ExperimentConfig(
        name=name.strip(),
        root=root,
        output_dir=resolve(raw.get("output_dir", "outputs"), "output_dir"),
        protocol=protocol,
        methods=dict(methods),
    )
