from utils import verify_backends


def test_official_backends_are_pinned_and_unmodified() -> None:
    results = verify_backends()
    assert len(results) == 5
    assert all(item["ok"] for item in results)
