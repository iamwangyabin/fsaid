from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from file_io import sha256_file


class BenchmarkError(RuntimeError):
    """Base error with an actionable user-facing message."""


class ConfigurationError(BenchmarkError):
    """Raised when an experiment configuration is invalid."""


class DataLeakageError(BenchmarkError):
    """Raised when support/query or source/target leakage is detected."""


class ProvenanceError(BenchmarkError):
    """Raised when integrated method code differs from its implementation lock."""


@dataclass(frozen=True)
class ImplementationLock:
    methods: tuple[str, ...]
    display_name: str
    reference_repository: str
    reference_commit: str
    file_hashes: dict[str, str]


IMPLEMENTATION_LOCKS = (
    ImplementationLock(
        methods=("fsd",),
        display_name="FSD",
        reference_repository="teheperinko541/Few-Shot-AIGI-Detector",
        reference_commit="b545c05f3c927ef67c1b00f9a8badf3b68c5f4b3",
        file_hashes={
            "methods/fsd.py": "cfaf77fd1305e3a74d79f8a23644a60534f049ea281dd635f866f846aa68be05",
            "genimage_arrow.py": "eb73ab51f0c5e1bbc440f38cf9299aaab92677a67b2b71562340a82ab184416e",
            "train_fsd.py": "d052ad5b1f8f34641a2853872a317e1599703aa92f12967a98014e1042ce44dc",
        },
    ),
    ImplementationLock(
        methods=("ftnet", "ftnet_t"),
        display_name="FTNet / FTNet-T",
        reference_repository="zuiluorenjian/FTNet",
        reference_commit="139348d3a7627160cdfb1e4f537986bdf3c007f4",
        file_hashes={
            "models.py": "0dff9bf26c16b754da20c621ff9f4fc9b8d0ac8fc0af2d04d761dfc2c1c65ce9",
            "methods/ftnet.py": "bebc3fbe3521d0bff1be1fac866b172b9bccc19068273d04aa6fc0017097f6de",
        },
    ),
    ImplementationLock(
        methods=("clipdet",),
        display_name="CLIPDet-eval",
        reference_repository="grip-unina/ClipBased-SyntheticImageDetection",
        reference_commit="c76ef7f5e158c5aba9e55b8b94ab0079720d281e",
        file_hashes={
            "methods/base.py": "62394cd07960aa5ccc9716d3a6ab14473a81f4e91ddf3ad29ce8eda5e4591337",
            "methods/clipdet.py": "dd3f0f2c48d1361aabb577b8c8c16c31e5575a4eeb38394c91b931d62bd3bb5c",
        },
    ),
    ImplementationLock(
        methods=("omnidfa_detection",),
        display_name="OmniDFA-Detection-eval",
        reference_repository="teheperinko541/OmniDFA",
        reference_commit="35b9052e83e05436682095818693493f79da9458",
        file_hashes={
            "methods/base.py": "62394cd07960aa5ccc9716d3a6ab14473a81f4e91ddf3ad29ce8eda5e4591337",
            "methods/omnidfa.py": "d3ee8d928fa2d850e855fdeab44b6e1ec5d7cdd34b83d25f1051fb469f3b48c2",
        },
    ),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parent


def verify_backends(
    root: Path | None = None,
    strict: bool = True,
    methods: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    repo = root or repository_root()
    selected = set(methods) if methods is not None else None
    results: list[dict[str, object]] = []
    failures = []
    for implementation in IMPLEMENTATION_LOCKS:
        if selected is not None and selected.isdisjoint(implementation.methods):
            continue
        hash_status = {}
        for relative_path, expected in implementation.file_hashes.items():
            path = repo / relative_path
            actual = sha256_file(path) if path.is_file() else None
            hash_status[relative_path] = actual == expected
        ok = all(hash_status.values())
        results.append(
            {
                "name": implementation.display_name,
                "methods": list(implementation.methods),
                "implementation": "integrated",
                "reference_repository": implementation.reference_repository,
                "reference_commit": implementation.reference_commit,
                "file_hashes": hash_status,
                "ok": ok,
            }
        )
        if not ok:
            failures.append(implementation.display_name)
    if failures and strict:
        raise ProvenanceError(
            "Integrated method files differ from the implementation lock: " + ", ".join(failures)
        )
    return results


def binary_metrics(
    labels: Sequence[int],
    fake_probabilities: Sequence[float],
    decision_threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(fake_probabilities, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError("labels and fake_probabilities must be one-dimensional with equal length")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty query set")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("labels must contain only 0 (real) and 1 (fake)")
    if not np.isfinite(y_score).all():
        raise ValueError("fake scores must all be finite")
    # torch.argmax([real, fake]) resolves an exact tie to class 0 (real).
    y_pred = (y_score > decision_threshold).astype(np.int64)
    has_both = np.unique(y_true).size == 2
    real = y_true == 0
    fake = y_true == 1
    real_accuracy = float((y_pred[real] == 0).mean()) if real.any() else math.nan
    fake_accuracy = float((y_pred[fake] == 1).mean()) if fake.any() else math.nan
    return {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (real_accuracy + fake_accuracy) / 2
        if has_both
        else math.nan,
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "average_precision": float(average_precision_score(y_true, y_score))
        if has_both
        else math.nan,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else math.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
