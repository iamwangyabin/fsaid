from pathlib import Path

import pytest
import torch

from data import Sample
from scripts.run_cached_ftnet import install_feature_cache


class DummyMethod:
    device = torch.device("cpu")


def test_installs_feature_lookup_by_sample_identity(tmp_path: Path) -> None:
    first = Sample(tmp_path / "first.png", 0, "stage")
    second = Sample(tmp_path / "second.png", 1, "stage")
    method = DummyMethod()
    install_feature_cache(
        method,
        [first.identity, second.identity],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    selected = method._extract([second, first])
    assert torch.equal(selected, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))


def test_feature_lookup_rejects_missing_samples(tmp_path: Path) -> None:
    present = Sample(tmp_path / "present.png", 0, "stage")
    missing = Sample(tmp_path / "missing.png", 1, "stage")
    method = DummyMethod()
    install_feature_cache(method, [present.identity], torch.tensor([[1.0]]))

    with pytest.raises(ValueError, match="missing sample"):
        method._extract([missing])
