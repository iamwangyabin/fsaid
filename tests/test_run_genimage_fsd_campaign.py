from pathlib import Path

from scripts.run_genimage_fsd_campaign import checkpoint_step, latest_checkpoint


def test_selects_latest_complete_named_checkpoint(tmp_path: Path) -> None:
    for name in (
        "resnet50_step_10000.pth",
        "resnet50_step_30000.pth",
        "resnet50_step_20000.pth",
        "resnet50_step_40000.pth.partial",
        "unrelated.pth",
    ):
        (tmp_path / name).touch()

    assert latest_checkpoint(tmp_path, total_steps=25_000) == (
        tmp_path / "resnet50_step_20000.pth"
    )
    assert checkpoint_step(tmp_path / "resnet50_step_20000.pth") == 20_000
    assert checkpoint_step(tmp_path / "resnet50_step_20000.pth.partial") is None


def test_returns_none_when_no_checkpoint_exists(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path, total_steps=200_000) is None
