import importlib.util

import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="torch is an optional method dependency")
def test_fsd_prototype_scores_are_negative_squared_euclidean() -> None:
    import torch

    from methods.fsd import prototypical_scores

    support = torch.tensor(
        [
            [[0.0, 0.0], [0.2, 0.0]],
            [[2.0, 2.0], [2.2, 2.0]],
        ]
    )
    query = torch.tensor([[0.1, 0.0], [2.1, 2.0]])
    scores = prototypical_scores(query, support)
    expected = torch.tensor([[-0.0, -8.0], [-8.0, -0.0]])
    assert torch.allclose(scores, expected, atol=1e-6)
