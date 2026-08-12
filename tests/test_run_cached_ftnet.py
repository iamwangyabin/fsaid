import importlib.util
from pathlib import Path

import pytest

from data import Sample


torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch is an optional dependency")


def _method():
    import torch

    from methods.ftnet import FTNetMethod

    method = FTNetMethod.__new__(FTNetMethod)
    method.device = torch.device("cpu")
    method.feature_positions = None
    method.cached_features = None
    return method


def test_installs_feature_lookup_by_sample_identity(tmp_path: Path) -> None:
    import torch

    first = Sample(tmp_path / "first.png", 0, "stage")
    second = Sample(tmp_path / "second.png", 1, "stage")
    method = _method()
    method.set_feature_cache(
        [first.identity, second.identity],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    selected = method._features([second, first])
    assert torch.equal(selected, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))


def test_feature_lookup_rejects_missing_samples(tmp_path: Path) -> None:
    import torch

    present = Sample(tmp_path / "present.png", 0, "stage")
    missing = Sample(tmp_path / "missing.png", 1, "stage")
    method = _method()
    method.set_feature_cache([present.identity], torch.tensor([[1.0]]))

    with pytest.raises(ValueError, match="missing sample"):
        method._features([missing])
