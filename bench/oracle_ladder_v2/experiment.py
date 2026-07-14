"""OL-002 namespace and rendering delta over the frozen OL-001 implementation.

The scientific harness remains in :mod:`bench.oracle_ladder.experiment` and is
source-hashed by OL-002.  This module temporarily supplies a new experiment/schema
namespace plus one deterministic CSV fix while a public operation is running.  The
context is process-local, synchronous, and always restores the frozen OL-001 module.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from bench.oracle_ladder import experiment as base

SCHEMA_VERSION = "oracle-ladder-v2"
EXPERIMENT_ID = "OL-002"
DEFAULT_OUTPUT = Path("bench/oracle_ladder_v2/results/OL-002")
PROTOCOL_DOC = Path("docs/research/2026-07-14-oracle-prefix-ladder-ol002-protocol.md")
FAILURE_DOC = Path("docs/research/2026-07-14-ol001-verifier-failure.md")
FAILED_OUTPUT = Path("bench/oracle_ladder/results/OL-001")

FAILED_HASHES = {
    "protocol.json": "e5a01edaacf0150853db3db2349edd31e0c3b5386ac8740ef2826d03e43f768e",
    "input-manifest.json": "a6a241d1fce5ce1a71120c42a4779f2d5e4d13006fab9b5a87422f3af0113696",
    "OL-001-results.json": "0f0273f9974288ad66368d38494d3bd83f06bc42e2dca3a2a65075b881982faa",
    "OL-001-runs.csv": "a490af7967a63e8e5f05a22452f722a0ae0fdb5855d155babf86b70593cc0b50",
    "OL-001-report.md": "cfb50325d56b80d49350a1cfbb5032ec0f9ae89722fcdb8d3e54af2d9d6867d6",
    "inputs/BC-001-b1_r1_d8.npz": "9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948",
    "artifact-manifest.json": "b6956d01a87db378883d382f83ba98c4e135fa4793de5f8d37dad25514cca75d",
}

INHERITED_SCIENTIFIC_FIELDS = (
    "runtime_constraints",
    "development_seed_excluded",
    "formal_model_seeds",
    "training",
    "planner",
    "evaluation",
    "endpoint_rungs",
    "conditional_prefix_rungs",
    "contrasts",
    "fixed_audit_bank",
    "thresholds",
)

SOURCE_FILES = (
    Path("bench/oracle_ladder/__init__.py"),
    Path("bench/oracle_ladder/audit.py"),
    Path("bench/oracle_ladder/experiment.py"),
    Path("bench/oracle_ladder/models.py"),
    Path("bench/oracle_ladder_v2/__init__.py"),
    Path("bench/oracle_ladder_v2/__main__.py"),
    Path("bench/oracle_ladder_v2/experiment.py"),
    Path("tests/test_oracle_ladder.py"),
    Path("tests/test_oracle_ladder_v2.py"),
    Path("bench/bridge_control/__init__.py"),
    Path("bench/bridge_control/__main__.py"),
    Path("bench/bridge_control/experiment.py"),
    Path("bench/bridge_control/fixture.py"),
    Path("bench/bridge_control/report.py"),
    Path("src/prospect/agent.py"),
    Path("src/prospect/planning.py"),
    Path("src/prospect/types.py"),
    Path("src/prospect/world_model.py"),
    Path("docs/research/2026-07-14-oracle-prefix-ladder-protocol.md"),
    PROTOCOL_DOC,
    FAILURE_DOC,
    base.PARENT_PROMPT,
    base.PARENT_PORTFOLIO,
)
OUTCOME_PATHS = (
    Path(f"{EXPERIMENT_ID}-results.json"),
    Path(f"{EXPERIMENT_ID}-runs.csv"),
    Path(f"{EXPERIMENT_ID}-report.md"),
    Path("artifact-manifest.json"),
)
ARTIFACT_PATHS = (
    Path("protocol.json"),
    Path("input-manifest.json"),
    base.INPUT_COPY,
    Path(f"{EXPERIMENT_ID}-results.json"),
    Path(f"{EXPERIMENT_ID}-runs.csv"),
    Path(f"{EXPERIMENT_ID}-report.md"),
)

_BASE_PROTOCOL_RECORD = base.protocol_record
_BASE_CSV_TEXT = base._csv_text
_BASE_REPORT_TEXT = base._report_text


def _failed_predecessor() -> dict[str, object]:
    failed_root = base.REPO_ROOT / FAILED_OUTPUT
    actual_hashes = {name: base._file_hash(failed_root / name) for name in FAILED_HASHES}
    if actual_hashes != FAILED_HASHES:
        raise ValueError("preserved OL-001 failed artifacts have drifted")
    failed_protocol = base._read_json(failed_root / "protocol.json")
    required_numpy = cast(dict[str, str], failed_protocol["runtime_constraints"])["numpy_version"]
    if base.np.__version__ != required_numpy:
        raise ValueError(f"OL-002 requires the OL-001 NumPy runtime {required_numpy}, got {base.np.__version__}")
    current_sources = {
        path: base._file_hash(base.REPO_ROOT / path) for path in cast(dict[str, str], failed_protocol["source_sha256"])
    }
    if current_sources != failed_protocol["source_sha256"]:
        raise ValueError("OL-001 source snapshot no longer matches its frozen protocol")
    failed_artifact_manifest = base._read_json(failed_root / "artifact-manifest.json")
    expected_manifest_artifacts = {
        name: digest for name, digest in FAILED_HASHES.items() if name != "artifact-manifest.json"
    }
    if failed_artifact_manifest.get("artifacts") != expected_manifest_artifacts:
        raise ValueError("OL-001 artifact manifest does not match the pinned complete package")
    failed_results = base._read_json(failed_root / "OL-001-results.json")
    actual_csv = (failed_root / "OL-001-runs.csv").read_bytes()
    canonical_csv = _BASE_CSV_TEXT(cast(list[dict[str, object]], failed_results["rows"]))
    if actual_csv != canonical_csv.encode("utf-8"):
        raise ValueError("OL-001 failure is not the recorded newline-only verifier defect")
    return {
        "experiment_id": "OL-001",
        "path": str(FAILED_OUTPUT),
        "terminal_failure": "CSV text comparison after universal-newline translation",
        "artifact_sha256": actual_hashes,
        "artifact_manifest_entries_verified": True,
        "source_snapshot_matches_protocol": True,
        "canonical_csv_bytes_match": True,
        "csv_crlf_count": actual_csv.count(b"\r\n"),
        "normalized_text_matches_crlf_canonical": (
            (failed_root / "OL-001-runs.csv").read_text(encoding="utf-8") == canonical_csv
        ),
        "failure_record": str(FAILURE_DOC),
        "failure_record_sha256": base._file_hash(base.REPO_ROOT / FAILURE_DOC),
    }


def _protocol_record_v2() -> dict[str, object]:
    record = _BASE_PROTOCOL_RECORD()
    failed_protocol = base._read_json(base.REPO_ROOT / FAILED_OUTPUT / "protocol.json")
    inherited = {field: record[field] for field in INHERITED_SCIENTIFIC_FIELDS}
    failed_inherited = {field: failed_protocol[field] for field in INHERITED_SCIENTIFIC_FIELDS}
    if inherited != failed_inherited:
        raise ValueError("OL-002 scientific protocol or runtime differs from frozen OL-001")
    record["stop_rules"] = [
        str(rule).replace("OL-001", EXPERIMENT_ID) for rule in cast(list[str], record["stop_rules"])
    ]
    record["status"] = "frozen_administrative_rerun_after_verifier_defect"
    record["failed_predecessor"] = _failed_predecessor()
    record["method_delta"] = {
        "scientific_changes": [],
        "administrative": {"experiment_id": EXPERIMENT_ID, "schema_version": SCHEMA_VERSION},
        "rendering": "canonical CSV CRLF row terminators converted to LF before write and compare",
        "epistemic_role": (
            "administrative full rerun frozen after OL-001 outcomes; matching results are not "
            "independent evidence and must not be double-counted"
        ),
        "inherited_scientific_protocol_sha256": base.sha256(base._canonical_json_bytes(failed_inherited)).hexdigest(),
    }
    return record


def _csv_text_v2(rows: list[dict[str, object]]) -> str:
    return _BASE_CSV_TEXT(rows).replace("\r\n", "\n")


def _report_text_v2(results: dict[str, Any]) -> str:
    return _BASE_REPORT_TEXT(results).replace("OL-001", EXPERIMENT_ID)


@contextmanager
def _configured() -> Iterator[None]:
    patches: dict[str, object] = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
        "PROTOCOL_DOC": PROTOCOL_DOC,
        "SOURCE_FILES": SOURCE_FILES,
        "OUTCOME_PATHS": OUTCOME_PATHS,
        "ARTIFACT_PATHS": ARTIFACT_PATHS,
        "protocol_record": _protocol_record_v2,
        "_csv_text": _csv_text_v2,
        "_report_text": _report_text_v2,
    }
    original = {name: getattr(base, name) for name in patches}
    try:
        for name, value in patches.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    with _configured():
        return base.prepare(output)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    with _configured():
        return base.run(output)


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
) -> dict[str, Any]:
    with _configured():
        return base.verify(output, require_results=require_results)


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    with _configured():
        return base.analyze(output)
