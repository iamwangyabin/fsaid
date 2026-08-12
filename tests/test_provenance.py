from utils import verify_backends


def test_official_backends_are_pinned_and_unmodified() -> None:
    results = verify_backends()
    assert len(results) == 4
    assert all(item["ok"] for item in results)


def test_backend_verification_can_target_selected_methods() -> None:
    results = verify_backends(methods=("ftnet_t",))
    assert [item["name"] for item in results] == ["FTNet / FTNet-T"]
