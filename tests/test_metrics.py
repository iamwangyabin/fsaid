import pytest

from utils import binary_metrics


def test_binary_metrics_perfect() -> None:
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["f1"] == 1.0


def test_binary_metrics_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        binary_metrics([0], [0.1, 0.2])


def test_binary_metrics_supports_native_score_thresholds() -> None:
    result = binary_metrics([0, 1], [0.2, -0.2], decision_threshold=0.0)
    assert result["accuracy"] == 0.0
    inverted = binary_metrics([0, 1], [-0.2, 0.2], decision_threshold=0.0)
    assert inverted["accuracy"] == 1.0


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 2], [0.1, 0.9], "labels"),
        ([0, 1], [0.1, float("nan")], "finite"),
    ],
)
def test_binary_metrics_rejects_invalid_values(labels, scores, message) -> None:
    with pytest.raises(ValueError, match=message):
        binary_metrics(labels, scores)
