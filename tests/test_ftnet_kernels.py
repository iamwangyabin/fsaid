import importlib.util

import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="torch is an optional method dependency")
def test_ftnet_cache_matches_equations() -> None:
    import torch

    from methods.ftnet import build_cache, cache_logits

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    keys, values = build_cache(features, labels)
    logits = cache_logits(features, keys, values, alpha=15.0)
    tiny = torch.exp(torch.tensor(-15.0))
    expected = torch.tensor([[1.0, tiny], [tiny, 1.0]])
    assert torch.allclose(logits, expected)
