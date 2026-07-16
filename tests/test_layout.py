from pathlib import Path


def test_methods_are_integrated_not_external_repositories() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / ".gitmodules").exists()
    assert not (root / "external").exists()
    assert not (root / "src").exists()
    assert (root / "methods/fsd.py").is_file()
    assert (root / "methods/ftnet.py").is_file()
    assert (root / "methods/clipdet.py").is_file()
    assert (root / "methods/omnidfa.py").is_file()


def test_method_modules_do_not_dynamically_load_vendor_code() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "methods/fsd.py",
            "methods/ftnet.py",
            "methods/clipdet.py",
            "methods/omnidfa.py",
        )
    )
    assert "importlib" not in source
    assert "sys.path" not in source
    assert "subprocess" not in source
