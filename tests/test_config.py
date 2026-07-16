from pathlib import Path

import pytest

from config import load_config
from utils import ConfigurationError


def _write_config(path: Path, query_per_class: str = "null") -> None:
    path.write_text(
        "\n".join(
            [
                "protocol:",
                "  manifest: manifest.csv",
                "  stages: [g1]",
                "  shots: [0]",
                "  seeds: [0]",
                f"  query_per_class: {query_per_class}",
                "methods:",
                "  clipdet: {}",
            ]
        ),
        encoding="utf-8",
    )


def test_zero_shot_config_is_valid_for_later_evaluation_only_selection(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path)
    assert load_config(path).protocol.shots == (0,)


def test_nonpositive_query_count_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path, "0")
    with pytest.raises(ConfigurationError, match="query_per_class"):
        load_config(path)


@pytest.mark.parametrize("field", ["shots", "seeds"])
def test_duplicate_plan_dimensions_are_rejected(tmp_path: Path, field: str) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(f"  {field}: [0]", f"  {field}: [0, 0]"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match=f"protocol.{field} contains duplicates"):
        load_config(path)


def test_string_boolean_is_not_silently_treated_as_true(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  query_per_class: null", "  query_per_class: null\n  cumulative_cache: 'false'"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="cumulative_cache"):
        load_config(path)
