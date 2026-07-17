from __future__ import annotations

import stat
from pathlib import Path

import pytest

from bench.multimodal_causal_assay import lcv_parent


def test_live_lcv_parent_is_the_exact_completed_authority() -> None:
    snapshot = lcv_parent.verify(lcv_parent.LIVE_ROOT)

    assert len(snapshot.files) == lcv_parent.FILE_COUNT == 42
    assert len(snapshot.directories) + 1 == lcv_parent.DIRECTORY_COUNT == 17
    assert sum(item.source.bytes for item in snapshot.files) == lcv_parent.TOTAL_BYTES == 7_821_446
    assert set(snapshot.records) == set(lcv_parent.EXPECTED_FILES)
    assert lcv_parent.parent_root(lcv_parent.LIVE_ROOT) == (
        lcv_parent.LIVE_ROOT / "prepared/inputs/MM-007"
    )


def test_lcv_parent_copy_is_complete_sealed_and_exclusive(tmp_path: Path) -> None:
    destination = tmp_path / "LCV-001"

    copied = lcv_parent.copy(lcv_parent.LIVE_ROOT, destination)

    assert len(copied.files) == 42
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert all(
        stat.S_IMODE((destination / relative).stat().st_mode) == 0o444
        for relative in lcv_parent.EXPECTED_FILES
    )
    with pytest.raises(lcv_parent.LCVParentError, match="already exists"):
        lcv_parent.copy(lcv_parent.LIVE_ROOT, destination)


def test_lcv_parent_rejects_post_copy_byte_mutation(tmp_path: Path) -> None:
    destination = tmp_path / "LCV-001"
    lcv_parent.copy(lcv_parent.LIVE_ROOT, destination)
    relative = lcv_parent.RESULT_RELATIVE
    target = destination / relative
    target.chmod(0o644)
    payload = bytearray(target.read_bytes())
    payload[-2] ^= 1
    target.write_bytes(payload)
    target.chmod(0o444)

    with pytest.raises(lcv_parent.LCVParentError, match="file hash differs"):
        lcv_parent.verify(destination)


def test_lcv_parent_rejects_post_copy_membership_growth(tmp_path: Path) -> None:
    destination = tmp_path / "LCV-001"
    lcv_parent.copy(lcv_parent.LIVE_ROOT, destination)
    destination.chmod(0o755)
    extra = destination / "unexpected.txt"
    extra.write_text("not authoritative\n", encoding="utf-8")
    extra.chmod(0o444)
    destination.chmod(0o555)

    with pytest.raises(lcv_parent.LCVParentError, match="tree membership differs"):
        lcv_parent.verify(destination)


def test_lcv_parent_module_has_no_historical_experiment_import() -> None:
    source = Path(lcv_parent.__file__).read_text(encoding="utf-8")

    assert "multimodal_resolution_diagnostics" not in source
    assert "multimodal_causal_diagnostics" not in source
    assert "multimodal_mechanism_diagnostics" not in source
