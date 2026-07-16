#!/usr/bin/env python3
"""Run FTNet continual plans with a verified, reusable frozen-feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig, load_config
from data import Sample, load_manifest
from methods.base import FewShotMethod
from run import _create_method, _run_plan, _write_results, validate_experiment
from utils import verify_backends


METHODS = ("ftnet", "ftnet_t")


def feature_signature(
    config: ExperimentConfig, samples: Sequence[Sample], method_name: str
) -> str:
    method_config = config.methods[method_name]
    digest = hashlib.sha256()
    digest.update(b"fsaid-ftnet-feature-cache-v1\0")
    digest.update(str(method_config.get("backbone", "ViT-L/14")).encode())
    digest.update(b"\0")
    digest.update(str(method_config.get("clip_layer", 12)).encode())
    for sample in samples:
        digest.update(b"\0")
        digest.update(sample.identity.encode())
    return digest.hexdigest()


def install_feature_cache(
    method: FewShotMethod,
    identities: Sequence[str],
    features: torch.Tensor,
) -> None:
    if features.ndim != 2 or features.shape[0] != len(identities):
        raise ValueError("Feature cache dimensions do not match its identity list")
    if len(set(identities)) != len(identities):
        raise ValueError("Feature cache identity list contains duplicates")
    positions = {identity: index for index, identity in enumerate(identities)}

    def cached_extract(self, selected: Sequence[Sample]) -> torch.Tensor:
        try:
            indices = [positions[sample.identity] for sample in selected]
        except KeyError as exc:
            raise ValueError(f"Feature cache is missing sample {exc.args[0]}") from exc
        return features[indices].to(self.device)

    method._extract = MethodType(cached_extract, method)  # type: ignore[attr-defined]


def build_or_load_cache(
    method: FewShotMethod,
    config: ExperimentConfig,
    samples: Sequence[Sample],
    cache_path: Path,
    extraction_chunk_size: int,
    signature_method: str,
) -> tuple[list[str], torch.Tensor]:
    signature = feature_signature(config, samples, signature_method)
    cache_path = cache_path.expanduser().resolve()
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError(f"Feature cache signature mismatch: {cache_path}")
        identities = payload.get("identities")
        features = payload.get("features")
        if not isinstance(identities, list) or not isinstance(features, torch.Tensor):
            raise ValueError(f"Invalid feature cache payload: {cache_path}")
        return identities, features

    outputs = []
    for offset in range(0, len(samples), extraction_chunk_size):
        chunk = samples[offset : offset + extraction_chunk_size]
        extracted = method._extract(chunk)  # type: ignore[attr-defined]
        outputs.append(extracted.detach().cpu())
        print(
            json.dumps(
                {
                    "cache": str(cache_path),
                    "extracted": min(offset + extraction_chunk_size, len(samples)),
                    "total": len(samples),
                }
            ),
            flush=True,
        )
    features = torch.cat(outputs, dim=0)
    identities = [sample.identity for sample in samples]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".partial")
    torch.save(
        {
            "format_version": 1,
            "signature": signature,
            "identities": identities,
            "features": features,
        },
        temporary,
    )
    os.replace(temporary, cache_path)
    return identities, features


def run_cached(
    config_path: Path,
    selected_methods: Sequence[str],
    cache_path: Path,
    extraction_chunk_size: int = 1024,
) -> list[dict[str, Any]]:
    verify_backends()
    config = load_config(config_path)
    unknown = set(selected_methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unsupported cached methods: {sorted(unknown)}")
    if extraction_chunk_size <= 0:
        raise ValueError("extraction_chunk_size must be positive")
    plans = validate_experiment(config)
    manifest_samples = load_manifest(config.protocol.manifest)
    samples_by_identity = {sample.identity: sample for sample in manifest_samples}
    needed_identities = dict.fromkeys(
        sample.identity
        for plan in plans
        for stage in plan.stages
        for sample in (*stage.support, *stage.query)
    )
    samples = [samples_by_identity[identity] for identity in needed_identities]
    records: list[dict[str, Any]] = []
    cached_payload: tuple[list[str], torch.Tensor] | None = None

    for method_name in selected_methods:
        method = _create_method(method_name, config)
        try:
            if cached_payload is None:
                cached_payload = build_or_load_cache(
                    method,
                    config,
                    samples,
                    cache_path,
                    extraction_chunk_size,
                    method_name,
                )
            identities, features = cached_payload
            install_feature_cache(method, identities, features)
            for plan in plans:
                records.extend(_run_plan(config, plan, method))
        finally:
            method.close()

    _write_results(config.output_dir, records)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--extraction-chunk-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = METHODS if args.method == "all" else (args.method,)
    records = run_cached(
        args.config,
        selected,
        args.cache,
        extraction_chunk_size=args.extraction_chunk_size,
    )
    print(json.dumps(records, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
