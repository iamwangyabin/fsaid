import importlib.util
from types import SimpleNamespace

import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="torch is an optional method dependency")
def test_clip_layer_encoder_stops_after_the_requested_residual_block() -> None:
    import torch
    import torch.nn as nn

    from models import CLIPLayerEncoder

    class AddVector(nn.Module):
        def __init__(self, vector):
            super().__init__()
            self.register_buffer("vector", torch.tensor(vector, dtype=torch.float32))

        def forward(self, value):
            return value + self.vector

    visual = SimpleNamespace(
        conv1=nn.Conv2d(3, 2, kernel_size=2, stride=2, bias=False),
        class_embedding=torch.tensor([1.0, 0.0]),
        positional_embedding=torch.zeros(5, 2),
        ln_pre=nn.Identity(),
        transformer=SimpleNamespace(
            resblocks=nn.Sequential(AddVector([0.0, 1.0]), AddVector([1.0, 0.0]))
        ),
    )
    nn.init.zeros_(visual.conv1.weight)
    encoder = CLIPLayerEncoder.__new__(CLIPLayerEncoder)
    encoder.device = torch.device("cpu")
    encoder.layer = 1
    encoder.model = SimpleNamespace(visual=visual, dtype=torch.float32)

    result = encoder.encode(torch.zeros(1, 3, 4, 4))

    expected = torch.tensor([[2**-0.5, 2**-0.5]])
    assert torch.allclose(result, expected)
