#!/usr/bin/env python3
"""Run the supported cross-dataset benchmark matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_cross_benchmark import (
    PROJECT_ROOT,
    RUN_CONFIGS,
    expected_counts,
    results_complete,
    summarize,
)


MANIFEST_SOURCES = {
    "genimage": "data/manifests/genimage.csv",
    "opensdi": "data/manifests/opensdi.csv",
    "synthbuster": "data/manifests/synthbuster_clipdet_official.csv",
}


def run_campaign(python: str = sys.executable) -> None:
    for dataset, source in MANIFEST_SOURCES.items():
        subprocess.run(
            (
                python,
                "scripts/prepare_cross_benchmark_manifest.py",
                "--input",
                source,
                "--output",
                f"data/manifests/cross_benchmark/{dataset}.csv",
            ),
            cwd=PROJECT_ROOT,
            check=True,
        )

    output_root = PROJECT_ROOT / "outputs" / "cross_benchmark"
    for dataset, families in RUN_CONFIGS.items():
        for family, config_path in families.items():
            result_path = output_root / dataset / family / "results.jsonl"
            expected = expected_counts(config_path)
            if results_complete(result_path, expected):
                print(json.dumps({"event": "skip", "dataset": dataset, "family": family}))
                continue
            if family == "fixed":
                command = (python, "run.py", "run", "--config", str(config_path))
            else:
                command = (
                    python,
                    "scripts/run_cached_ftnet.py",
                    "--config",
                    str(config_path),
                    "--cache",
                    f"outputs/feature_cache/cross_benchmark/{dataset}/ftnet.pt",
                )
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            if not results_complete(result_path, expected):
                raise RuntimeError(f"Incomplete results: {dataset}/{family}")

    summarize(output_root, require_complete=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    run_campaign(parse_args().python)


if __name__ == "__main__":
    main()
