import pytest
from PIL import Image
from types import SimpleNamespace


torch = pytest.importorskip("torch", reason="torch is an optional method dependency")

from methods.omnidfa import OmniValidationPreprocess  # noqa: E402


@pytest.mark.parametrize("size", [(100, 300), (300, 100), (128, 128)])
def test_omnidfa_validation_resize_makes_both_sides_at_least_256(size) -> None:
    output = OmniValidationPreprocess(256)(Image.new("RGB", size))
    assert min(output.size) == 256


def test_omnidfa_validation_resize_leaves_large_images_unchanged() -> None:
    image = Image.new("RGB", (300, 400))
    output = OmniValidationPreprocess(256)(image)
    assert output.size == image.size


def test_omnidfa_reports_the_released_real_positive_metrics() -> None:
    from methods.omnidfa import OmniDFADetectionMethod

    method = OmniDFADetectionMethod.__new__(OmniDFADetectionMethod)
    method.center = SimpleNamespace(cosine_threshold=torch.tensor(0.5))
    metrics = method.official_metrics(
        labels=[0, 0, 1, 1],
        fake_scores=[-0.9, -0.8, -0.2, -0.1],
    )
    assert metrics["official_balanced_accuracy"] == 1.0
    assert metrics["official_real_positive_average_precision_20"] == 1.0
