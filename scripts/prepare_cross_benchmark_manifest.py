#!/usr/bin/env python3
"""Create a fixed-query manifest for cross-dataset few-shot comparisons."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import MANIFEST_COLUMNS, Sample, load_manifest, stable_order, write_manifest
from file_io import sha256_file, write_json
from utils import ConfigurationError


FORMAT_VERSION = 1


def prepare_manifest(
    input_path: Path,
    output_path: Path,
    support_candidates_per_class: int = 30,
    reserve_seed: int = 20260812,
) -> dict[str, object]:
    if support_candidates_per_class <= 0:
        raise ConfigurationError("support_candidates_per_class must be positive")

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    samples = load_manifest(input_path)
    non_pool = [sample for sample in samples if sample.split != "pool"]
    if non_pool:
        raise ConfigurationError(
            "Cross-benchmark input must contain only pool rows; "
            f"found {len(non_pool)} explicit rows"
        )

    support_identities: set[str] = set()
    generators = tuple(dict.fromkeys(sample.generator for sample in samples))
    for generator in generators:
        for label in (0, 1):
            candidates = [
                sample
                for sample in samples
                if sample.generator == generator and sample.label == label
            ]
            if len(candidates) <= support_candidates_per_class:
                raise ConfigurationError(
                    f"Stage {generator}, label {label}: need more than "
                    f"{support_candidates_per_class} rows"
                )
            ordered = stable_order(
                candidates,
                reserve_seed,
                f"cross-benchmark:{generator}:{label}:reserve",
            )
            support_identities.update(
                sample.identity for sample in ordered[:support_candidates_per_class]
            )

    prepared = [
        Sample(
            path=sample.path,
            label=sample.label,
            generator=sample.generator,
            split="support" if sample.identity in support_identities else "query",
            sample_id=sample.sample_id,
        )
        for sample in samples
    ]
    temporary = output_path.with_name(f"{output_path.name}.partial-{os.getpid()}")
    write_manifest(prepared, temporary)
    os.replace(temporary, output_path)

    counts = Counter((sample.generator, sample.label, sample.split) for sample in prepared)
    summary: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "input_manifest": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_manifest": str(output_path),
        "output_sha256": sha256_file(output_path),
        "columns": list(MANIFEST_COLUMNS),
        "reserve_seed": reserve_seed,
        "support_candidates_per_class": support_candidates_per_class,
        "rows": len(prepared),
        "generators": list(generators),
        "counts": [
            {
                "generator": generator,
                "label": label,
                "split": split,
                "count": count,
            }
            for (generator, label, split), count in sorted(counts.items())
        ],
    }
    summary_path = output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--support-candidates-per-class", type=int, default=30)
    parser.add_argument("--reserve-seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_manifest(
        args.input,
        args.output,
        support_candidates_per_class=args.support_candidates_per_class,
        reserve_seed=args.reserve_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
