"""Exact copied-LCV custody for MM-011.

The complete LCV-001 package is the only parent authority.  This module pins every
file independently of the live LCV manifest, copies authenticated snapshot bytes into
the MM-011 tree, and then invokes only LCV's own structural verifier.  It never imports
an MM-001--MM-007 experiment or verifier.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from bench.sealed_lineage_verifier import custody
from bench.sealed_lineage_verifier import experiment as lcv_experiment

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
LIVE_ROOT: Final = REPO_ROOT / "bench/sealed_lineage_verifier/results/LCV-001"
PARENT_RELATIVE: Final = Path("prepared/inputs/MM-007")
RESULT_RELATIVE: Final = Path("outcomes/LCV-001-results.json")
PARENT_CLOSURE_RELATIVE: Final = Path("outcomes/parent-closure.json")
RUNTIME_RECEIPT_RELATIVE: Final = Path("outcomes/runtime-receipt.json")
CLEANUP_RECEIPT_RELATIVE: Final = Path("outcomes/cleanup-receipt.json")
ARTIFACT_MANIFEST_RELATIVE: Final = Path("outcomes/artifact-manifest.json")
FILE_COUNT: Final = 42
DIRECTORY_COUNT: Final = 17
TOTAL_BYTES: Final = 7_821_446


def _record(sha256: str, size: int) -> custody.ExpectedFile:
    return custody.ExpectedFile(sha256=sha256, bytes=size, mode=0o444)


EXPECTED_FILES: Final[Mapping[Path, custody.ExpectedFile]] = MappingProxyType(
    {
        Path("outcomes/LCV-001-report.md"): _record(
            "b945f1ce89854672296cfe32739cc06a9e684114b767ec18c3e7348f55801807", 906
        ),
        Path("outcomes/LCV-001-results.json"): _record(
            "d7e403008239d57b7f662f28cdb686a851f3cdd3c7164be4ff423bdeeabe1e94", 1_644
        ),
        Path("outcomes/artifact-manifest.json"): _record(
            "1ea64a430143315f9dd64c4f211a01d418d3a568d874cdef7d62943b27d0de5e", 7_880
        ),
        Path("outcomes/cleanup-receipt.json"): _record(
            "60f8c1f97f560d6c83ed8fff22183dd3528d60233b55fa20e37ff6acc3f0bce3", 1_067
        ),
        Path("outcomes/formal-start.json"): _record(
            "4757d6fe6584177bc9d3e4f66773ba04bc625d7e59ea77394afabdb968fb6d7f", 955
        ),
        Path("outcomes/mutation-controls.json"): _record(
            "2eb640ac9991c6543dd9846ac0781eec6577af061dabf0dd74b69f578e8048d5", 4_699
        ),
        Path("outcomes/parent-closure.json"): _record(
            "bf47ef0e18c13885c943d29939723791a1084c28b66572b876723f4ddf29d005", 7_923
        ),
        Path("outcomes/prepared-phase-anchor.json"): _record(
            "47d36c0b6da4fe279dd9f01653d5b70b5e71ee3d3809b3fc4f68abf2eb727f44", 133
        ),
        Path("outcomes/provisional-result.json"): _record(
            "bc17e27c235c947fd94e705260816f381cf29adc90785815cccbd28fc73d0390", 1_562
        ),
        Path("outcomes/runtime-receipt.json"): _record(
            "29df8291ddaf0f6e93c562686a6ad678716fd9d341d7508c55f725a818ee25b1", 306_477
        ),
        Path("prepared/LCV-001-protocol.md"): _record(
            "4fd0eb48cc5f9f9b49426a4dfdcab92f9c585664879aa6fff6c5969ed9bdbb6a", 13_593
        ),
        Path("prepared/config.json"): _record(
            "550d40a70850197592596e1d8608e7180c81ef4ba566fde88e6d56f94950c8cc", 15_712
        ),
        Path("prepared/freeze-record.json"): _record(
            "55a5ed39f70667c75fe8aa309cd5214f15b1e4a08a8857c42a69796518f01b36", 773
        ),
        Path("prepared/input-manifest.json"): _record(
            "6ca0ac97b73130673ed1f38dbeafd5b5f772026b4e2de87bf465fb07f887a1cd", 14_827
        ),
        Path("prepared/inputs/MM-007/MM-007-evidence.json"): _record(
            "13dfa89e541e6122263ea9814d42fb328da303dcc74556cdaaa5d5860d99abaf", 273_804
        ),
        Path("prepared/inputs/MM-007/MM-007-frames-64x64.npz"): _record(
            "fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f", 2_525_160
        ),
        Path("prepared/inputs/MM-007/MM-007-protocol.md"): _record(
            "24bbac1855cc2b51d2a65012b9c63037637c53555b86bbad7c66a6249108a73c", 15_238
        ),
        Path("prepared/inputs/MM-007/MM-007-report.md"): _record(
            "b18760128941ab2eff893b8c0afc469b92f71077d489e060d56519407990b8a2", 1_086
        ),
        Path("prepared/inputs/MM-007/MM-007-results.json"): _record(
            "3c92729e1e5c18c14461e36602bdb86acd31750d9f5a85f535cd33a43fb9c47b", 221_177
        ),
        Path("prepared/inputs/MM-007/artifact-manifest.json"): _record(
            "db0b6654ab098dc9a3ec93e4a6de8820bbe5860d44974645e9a5ee7dad1537fb", 2_678
        ),
        Path("prepared/inputs/MM-007/formal-start.json"): _record(
            "ea5c7bda870d71ead3172c1fc6e504d6a6b02d2ba785e9fd2fc75a91c667eee3", 2_090
        ),
        Path("prepared/inputs/MM-007/input-manifest.json"): _record(
            "1f83c805e6c5d75f4f1d5a2102d471c15bbc6bb787960cb5ae630bd2260faa1f", 43_618
        ),
        Path("prepared/inputs/MM-007/inputs/MM-004/input-manifest.json"): _record(
            "597a8bfc9f6ae1f6ff1f0d3be456f57d768f1866a1fac59cd981dd260076dc90", 31_328
        ),
        Path("prepared/inputs/MM-007/inputs/MM-006/MM-006-evidence.json"): _record(
            "5c5ffa514ab0f0c06c8588e69b54d6c4f2f6be3a4471a0fe7a31aa1e1dd3dac2", 3_350_469
        ),
        Path("prepared/inputs/MM-007/inputs/MM-006/MM-006-results.json"): _record(
            "c5e0737acf6030315a77b497f5d5ea78693eb8a5879399e37bdfb702e2b9f648", 208_329
        ),
        Path("prepared/inputs/MM-007/inputs/MM-006/artifact-manifest.json"): _record(
            "9727eefc6c5665b5eb8cc65ae9cfab57bb4c8b3e353b747363bec5e3c2f573b0", 2_381
        ),
        Path("prepared/inputs/MM-007/inputs/MM-006/input-manifest.json"): _record(
            "badd7676f1e4a60c56b59af12a1d7f82ef134e797febc86a5adcb4c33dda5cd1", 24_723
        ),
        Path("prepared/inputs/MM-007/inputs/MM-006/inputs/MM-005/inputs/MM-004/MM-004-pixel-grids.npz"): _record(
            "cca261a941e68a7ddc510eee3a3af958d33b6abaf958cb5562ed6b66c22f47c8", 409_427
        ),
        Path("prepared/pre-real-audit.json"): _record(
            "69cd2d3908f387c87efcbcb22e6a5051bfc5961886ee771a74ad76c59e336d31", 825
        ),
        Path("prepared/runtime/source/bench/__init__.py"): _record(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0
        ),
        Path("prepared/runtime/source/bench/sealed_lineage_verifier/__init__.py"): _record(
            "95820bcfda16e5c564a3aba5cf1b5f3563cdf0f8c4ab81099a36890a2cbbd447", 177
        ),
        Path("prepared/runtime/source/bench/sealed_lineage_verifier/__main__.py"): _record(
            "8e22401983860d5bffb2337a71a0eed191df6e6542650492074a0e22d8ddd036", 151
        ),
        Path("prepared/runtime/source/bench/sealed_lineage_verifier/bootstrap.py"): _record(
            "0b546146321fb6b3667cfb3e55e6baa81c560514d00204dc5fcd78353bc3a200", 782
        ),
        Path("prepared/runtime/source/bench/sealed_lineage_verifier/canary_probe.py"): _record(
            "50a6367b08a9852b646032f5103f22fb7ef1a66fa93fb757d184dfb109822c22", 2_423
        ),
        Path("prepared/runtime/source/bench/sealed_lineage_verifier/custody.py"): _record(
            "a06d668748c170ee2427ffca8158c6b1910cad40bbf233faae3d372680aa053b", 38_602
        ),
        Path("prepared/runtime/source/bench/sealed_lineage_verifier/experiment.py"): _record(
            "dd4dfa3b7c545663587ade6ca532b58942019b5268df853d2314942be1efe61f", 137_770
        ),
        Path("prepared/runtime/source/bench/sealed_lineage_verifier/runtime_probe.py"): _record(
            "31e036e5bff0fc8898807e18c9ecda94d1318d8780aa83568fa88f1703ab272c", 21_630
        ),
        Path("prepared/runtime/source/bench/sealed_lineage_verifier/supervisor.py"): _record(
            "8622d4bd05c1a26821efdac5a0974fbe21da8007e93850d971d491b65b7ca9cc", 26_810
        ),
        Path("prepared/runtime/source/tests/test_lcv001_custody.py"): _record(
            "873bb7805477d67e7b3d8e9ede3abf57e39ea5eb753274b90b1afbef2b550c37", 15_924
        ),
        Path("prepared/runtime/source/tests/test_lcv001_experiment.py"): _record(
            "c829009642c4042c658500a7fe31dcdb4a8239bb5000d9eda989129c2ea945c5", 64_451
        ),
        Path("prepared/runtime/source/tests/test_lcv001_runtime.py"): _record(
            "55a7634b2a2e59ab89b3d0458c2c735910e10ad4c7b346ceef4b0f923067605c", 2_090
        ),
        Path("prepared/runtime/source/tests/test_lcv001_supervisor.py"): _record(
            "e73836e751811b36548e9c2751da133ffa5310b33bbdf4f812035c43df8ab489", 20_152
        ),
    }
)


class LCVParentError(ValueError):
    """The live or copied LCV package differs from the frozen MM-011 authority."""


def _json_payload(snapshot: custody.TreeSnapshot, relative: Path) -> dict[str, object]:
    try:
        value = json.loads(snapshot.payloads[relative])
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LCVParentError(f"LCV JSON differs: {relative}") from error
    if type(value) is not dict:
        raise LCVParentError(f"LCV JSON root differs: {relative}")
    return cast(dict[str, object], value)


def _validate_summary(snapshot: custody.TreeSnapshot) -> None:
    if len(snapshot.files) != FILE_COUNT or len(snapshot.directories) + 1 != DIRECTORY_COUNT:
        raise LCVParentError("LCV package census differs")
    if sum(item.source.bytes for item in snapshot.files) != TOTAL_BYTES:
        raise LCVParentError("LCV package byte count differs")
    result = _json_payload(snapshot, RESULT_RELATIVE)
    if (
        result.get("experiment_id") != "LCV-001"
        or result.get("classification") != "PASS"
        or result.get("status") != "completed_after_cgroup_cleanup"
        or result.get("statement")
        != "host-bound sealed lineage/runtime verified after formal cgroup cleanup; no scientific outcome"
    ):
        raise LCVParentError("LCV result does not authorize the bounded successor surface")


def verify(root: Path) -> custody.TreeSnapshot:
    """Verify exact bytes/modes plus LCV's own completed-package semantics."""

    try:
        before = custody.verify_sealed_tree(root, EXPECTED_FILES)
        verified = lcv_experiment.verify(root)
        after = custody.verify_sealed_tree(root, EXPECTED_FILES)
    except (custody.CustodyError, lcv_experiment.InvalidLCV001Artifact) as error:
        raise LCVParentError(str(error)) from error
    if before.records != after.records or before.payloads != after.payloads:
        raise LCVParentError("LCV package changed during structural verification")
    if verified != {
        "classification": "PASS",
        "experiment_id": "LCV-001",
        "outcomes": "verified_results",
        "status": "verified",
    }:
        raise LCVParentError("LCV structural verifier result differs")
    _validate_summary(after)
    return after


def copy(source: Path, destination: Path) -> custody.TreeSnapshot:
    """Copy one descriptor-authenticated LCV snapshot and verify the sealed copy."""

    try:
        snapshot = custody.verify_sealed_tree(source, EXPECTED_FILES)
        _validate_summary(snapshot)
        custody.write_snapshot_exclusive(destination, snapshot)
    except custody.CustodyError as error:
        raise LCVParentError(str(error)) from error
    return verify(destination)


def parent_root(root: Path) -> Path:
    return root / PARENT_RELATIVE


__all__ = [
    "ARTIFACT_MANIFEST_RELATIVE",
    "CLEANUP_RECEIPT_RELATIVE",
    "DIRECTORY_COUNT",
    "EXPECTED_FILES",
    "FILE_COUNT",
    "LCVParentError",
    "LIVE_ROOT",
    "PARENT_CLOSURE_RELATIVE",
    "PARENT_RELATIVE",
    "RESULT_RELATIVE",
    "RUNTIME_RECEIPT_RELATIVE",
    "TOTAL_BYTES",
    "copy",
    "parent_root",
    "verify",
]
