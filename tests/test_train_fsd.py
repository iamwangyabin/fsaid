from pathlib import Path

import pytest

from train_fsd import train_fsd
from utils import ConfigurationError


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_steps": 0}, "total_steps"),
        ({"task_batch_size": 0}, "task_batch_size"),
        ({"workers": -1}, "workers"),
        ({"save_interval": 0}, "save_interval"),
        ({"accumulation_steps": 0}, "accumulation_steps"),
        ({"log_interval": 0}, "log_interval"),
        ({"data_format": "unsupported"}, "data_format"),
    ],
)
def test_invalid_training_counts_fail_before_loading_optional_dependencies(
    tmp_path: Path, kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        train_fsd(tmp_path, tmp_path / "output", "SD", **kwargs)
