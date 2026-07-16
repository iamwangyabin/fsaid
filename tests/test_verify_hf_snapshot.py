import hashlib
import json
from pathlib import Path

from scripts.verify_hf_snapshot import verify_snapshot


def test_verifies_lfs_files_from_hfd_metadata(tmp_path: Path) -> None:
    payload = b"fixed snapshot payload"
    data_path = tmp_path / "data" / "part.parquet"
    data_path.parent.mkdir()
    data_path.write_bytes(payload)
    metadata_path = tmp_path / ".hfd" / "repo_metadata.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text(
        json.dumps(
            {
                "sha": "fixed",
                "siblings": [
                    {
                        "rfilename": "data/part.parquet",
                        "lfs": {
                            "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = verify_snapshot(tmp_path, expected_revision="fixed")

    assert result["ok"] is True
    assert result["verified_lfs_files"] == 1
    assert result["verified_bytes"] == len(payload)
    assert result["failures"] == []
