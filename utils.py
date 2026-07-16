from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


class BenchmarkError(RuntimeError):
    """Base error with an actionable user-facing message."""


class ConfigurationError(BenchmarkError):
    """Raised when an experiment configuration is invalid."""


class DataLeakageError(BenchmarkError):
    """Raised when support/query or source/target leakage is detected."""


class ProvenanceError(BenchmarkError):
    """Raised when integrated method code differs from its implementation lock."""


@dataclass(frozen=True)
class IntegratedMethod:
    name: str
    reference_repository: str
    reference_commit: str
    file_hashes: dict[str, str]


METHODS = (
    IntegratedMethod(
        name="FSD",
        reference_repository="teheperinko541/Few-Shot-AIGI-Detector",
        reference_commit="b545c05f3c927ef67c1b00f9a8badf3b68c5f4b3",
        file_hashes={
            "methods/fsd.py": "470fdb342722972a76791efc1bff64a7585c4233df9c05222ee036b45572db24",
            "genimage_arrow.py": "b8eec347657587b59fd199d131637eda7cf42d3241f9fde4a56c78e149edd914",
            "train_fsd.py": "1a796c0f36c3494c1664f9d58815726919b6eb3a5ffd9f1dd04c693db36ec42d",
        },
    ),
    IntegratedMethod(
        name="FTNet",
        reference_repository="zuiluorenjian/FTNet",
        reference_commit="139348d3a7627160cdfb1e4f537986bdf3c007f4",
        file_hashes={
            "models.py": "0dff9bf26c16b754da20c621ff9f4fc9b8d0ac8fc0af2d04d761dfc2c1c65ce9",
            "methods/ftnet.py": "31673d84a4724c4993d5d9072a6651bd8a113c3e02340d703f4f7d2705542b80",
        },
    ),
    IntegratedMethod(
        name="FTNet-T",
        reference_repository="zuiluorenjian/FTNet",
        reference_commit="139348d3a7627160cdfb1e4f537986bdf3c007f4",
        file_hashes={
            "models.py": "0dff9bf26c16b754da20c621ff9f4fc9b8d0ac8fc0af2d04d761dfc2c1c65ce9",
            "methods/ftnet.py": "31673d84a4724c4993d5d9072a6651bd8a113c3e02340d703f4f7d2705542b80",
        },
    ),
    IntegratedMethod(
        name="CLIPDet-eval",
        reference_repository="grip-unina/ClipBased-SyntheticImageDetection",
        reference_commit="c76ef7f5e158c5aba9e55b8b94ab0079720d281e",
        file_hashes={
            "methods/clipdet.py": "90361d0c71c7836d91f22990fbc6dc0e76d3843c69b81e0cae372de555f63b06",
        },
    ),
    IntegratedMethod(
        name="OmniDFA-Detection-eval",
        reference_repository="teheperinko541/OmniDFA",
        reference_commit="35b9052e83e05436682095818693493f79da9458",
        file_hashes={
            "methods/omnidfa.py": "ce4d7892d3799ba078d2ac305f0437e950166b6cd827ce4aa909a0ea003f84bc",
        },
    ),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parent


def verify_backends(root: Path | None = None, strict: bool = True) -> list[dict[str, object]]:
    repo = root or repository_root()
    results: list[dict[str, object]] = []
    failures = []
    for method in METHODS:
        hash_status = {}
        for relative_path, expected in method.file_hashes.items():
            path = repo / relative_path
            actual = _sha256(path) if path.is_file() else None
            hash_status[relative_path] = expected != "TO_BE_LOCKED" and actual == expected
        ok = all(hash_status.values())
        results.append(
            {
                "name": method.name,
                "implementation": "integrated",
                "reference_repository": method.reference_repository,
                "reference_commit": method.reference_commit,
                "file_hashes": hash_status,
                "ok": ok,
            }
        )
        if not ok:
            failures.append(method.name)
    if failures and strict:
        raise ProvenanceError(
            "Integrated method files differ from the implementation lock: " + ", ".join(failures)
        )
    return results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
