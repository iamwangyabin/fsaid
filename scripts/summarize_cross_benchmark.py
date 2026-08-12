#!/usr/bin/env python3
"""Combine cross-benchmark outputs into method-by-dataset summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from file_io import write_json
from run import expected_result_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIGS = {
    dataset: {
        family: PROJECT_ROOT / "configs" / f"cross_benchmark_{dataset}_{family}.yaml"
        for family in ("fixed", "ftnet")
    }
    for dataset in ("genimage", "opensdi", "synthbuster")
}
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "average_precision",
    "roc_auc",
    "f1",
    "real_accuracy",
    "fake_accuracy",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def expected_counts(config_path: Path) -> dict[str, int]:
    return expected_result_counts(load_config(config_path))


def results_complete(path: Path, expected: dict[str, int]) -> bool:
    if not path.is_file():
        return False
    try:
        records = _read_jsonl(path)
    except (OSError, ValueError):
        return False
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get("method"))] += 1
    return dict(counts) == expected


def _mean(values: Iterable[Any]) -> float:
    numbers = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(numbers) / len(numbers) if numbers else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(root: Path, require_complete: bool = False) -> dict[str, Any]:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    status = []
    for dataset, families in RUN_CONFIGS.items():
        for family, config_path in families.items():
            path = root / dataset / family / "results.jsonl"
            expected = expected_counts(config_path)
            complete = results_complete(path, expected)
            status.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "results": str(path),
                    "complete": complete,
                    "expected": expected,
                }
            )
            if path.is_file():
                for record in _read_jsonl(path):
                    all_records.append({"dataset": dataset, **record})
    incomplete = [item for item in status if not item["complete"]]
    if require_complete and incomplete:
        missing = ", ".join(
            f"{item['dataset']}/{item['family']}" for item in incomplete
        )
        raise ValueError(f"Cross benchmark is incomplete: {missing}")

    matrix_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    stage_groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        key = (record["dataset"], record["method"], int(record["shots"]))
        matrix_groups[key].append(record)
        stage_groups[(*key, str(record["eval_generator"]))].append(record)

    matrix_rows = []
    for (dataset, method, shots), records in sorted(matrix_groups.items()):
        matrix_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "shots": shots,
                "records": len(records),
                **{
                    f"mean_{metric}": _mean(row.get(metric) for row in records)
                    for metric in METRICS
                },
            }
        )
    stage_rows = []
    for (dataset, method, shots, generator), records in sorted(stage_groups.items()):
        stage_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "shots": shots,
                "generator": generator,
                "seeds": len({int(row["seed"]) for row in records}),
                **{
                    f"mean_{metric}": _mean(row.get(metric) for row in records)
                    for metric in METRICS
                },
            }
        )

    _write_csv(root / "matrix.csv", matrix_rows)
    _write_csv(root / "by_generator.csv", stage_rows)
    payload = {
        "complete": not incomplete,
        "status": status,
        "records": len(all_records),
        "matrix_rows": len(matrix_rows),
    }
    write_json(root / "summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/cross_benchmark"))
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(summarize(args.root, args.require_complete), indent=2))


if __name__ == "__main__":
    main()
