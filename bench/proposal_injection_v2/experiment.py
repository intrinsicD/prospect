"""PI-002 namespace and canonical-report delta over frozen PI-001 source."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from bench.proposal_injection import experiment as base

SCHEMA_VERSION = "proposal-injection-v2"
EXPERIMENT_ID = "PI-002"
DEFAULT_OUTPUT = Path("bench/proposal_injection_v2/results/PI-002")
PROTOCOL_DOC = Path("docs/research/2026-07-14-proposal-injection-pi002-protocol.md")
FAILURE_DOC = Path("docs/research/2026-07-14-pi001-verifier-failure.md")
FAILED_OUTPUT = Path("bench/proposal_injection/results/PI-001")

FAILED_HASHES = {
    "protocol.json": "f2a8281175c6b083fc5b685359967c7e8297d46fe1f9404de6f6b84cc498c006",
    "input-manifest.json": "e8fa658fc9a6cc0e05a0ece9059a5635ca2ba828177669014dc69d2275fb443f",
    "PI-001-results.json": "92f4035c9f0247e10288ad0f412aeb1eb59922f835f7eaea027eefca88cfc409",
    "PI-001-runs.csv": "bb003b16ce91b918561381982ebd4e0c6eedae5c05ef3de59318a7bf8fd63c5e",
    "PI-001-report.md": "f622c0faf8d29e2a81088a791a4349e2ae5e9f1f1f26f41aedac5a75bf900c54",
    "inputs/BC-001-b1_r1_d8.npz": "9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948",
    "artifact-manifest.json": "f6d95ad83aad7d94b9c71665808b41ccc38c75487a104fbf61f077fd7f15842c",
}

SCIENTIFIC_FIELDS = (
    "formal_model_seeds",
    "development_seed_excluded",
    "training",
    "evaluation",
    "native_planner",
    "injection",
    "primary_arms",
    "conditional",
    "thresholds",
    "stop_rules",
)

SOURCE_FILES = (
    *base.SOURCE_FILES,
    Path("bench/proposal_injection_v2/__init__.py"),
    Path("bench/proposal_injection_v2/__main__.py"),
    Path("bench/proposal_injection_v2/experiment.py"),
    Path("tests/test_proposal_injection_v2.py"),
    PROTOCOL_DOC,
    FAILURE_DOC,
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
    *OUTCOME_PATHS[:-1],
)

_BASE_PROTOCOL_RECORD = base.protocol_record
_BASE_REPORT_TEXT = base._report_text


def _ordered_for_report(results: dict[str, Any]) -> dict[str, Any]:
    revised = dict(results)
    decision = dict(cast(dict[str, Any], results["decision"]))
    rescues = cast(dict[str, Any], decision["rescues"])
    decision["rescues"] = {name: rescues[name] for name in sorted(rescues)}
    revised["decision"] = decision
    return revised


def _report_text_v2(results: dict[str, Any]) -> str:
    return _BASE_REPORT_TEXT(_ordered_for_report(results)).replace(
        "# PI-001 proposal-injection result",
        "# PI-002 proposal-injection result",
        1,
    )


def _failed_predecessor() -> dict[str, object]:
    root = base.REPO_ROOT / FAILED_OUTPUT
    actual = {name: base._file_hash(root / name) for name in FAILED_HASHES}
    if actual != FAILED_HASHES:
        raise ValueError("preserved PI-001 failed artifacts have drifted")
    failed_protocol = base._read_json(root / "protocol.json")
    current_sources = {
        path: base._file_hash(base.REPO_ROOT / path)
        for path in cast(dict[str, str], failed_protocol["source_sha256"])
    }
    if current_sources != failed_protocol["source_sha256"]:
        raise ValueError("PI-001 frozen source snapshot has drifted")
    failed_manifest = base._read_json(root / "artifact-manifest.json")
    expected_manifest = {
        name: digest for name, digest in FAILED_HASHES.items() if name != "artifact-manifest.json"
    }
    if failed_manifest.get("artifacts") != expected_manifest:
        raise ValueError("PI-001 artifact manifest does not match the pinned package")

    failed_results = base._read_json(root / "PI-001-results.json")
    saved_report = (root / "PI-001-report.md").read_text(encoding="utf-8")
    serialized_order_report = _BASE_REPORT_TEXT(failed_results)
    original_order = dict(failed_results)
    original_decision = dict(cast(dict[str, Any], failed_results["decision"]))
    rescues = cast(dict[str, Any], original_decision["rescues"])
    preferred = [
        name
        for name in (
            "privileged_injection",
            "action_permuted_injection",
            "enlarged_native_search",
            "time_permuted_injection",
        )
        if name in rescues
    ]
    original_decision["rescues"] = {name: rescues[name] for name in preferred}
    original_order["decision"] = original_decision
    in_memory_order_report = _BASE_REPORT_TEXT(original_order)
    if saved_report != in_memory_order_report or saved_report == serialized_order_report:
        raise ValueError("PI-001 failure is not the recorded rescue-order report defect")
    first_difference = next(
        index
        for index, (saved, regenerated) in enumerate(
            zip(saved_report, serialized_order_report, strict=True)
        )
        if saved != regenerated
    )
    return {
        "experiment_id": "PI-001",
        "path": str(FAILED_OUTPUT),
        "terminal_failure": "canonical Markdown rescue order changed after sorted-key JSON serialization",
        "artifact_sha256": actual,
        "artifact_manifest_entries_verified": True,
        "source_snapshot_matches_protocol": True,
        "saved_report_matches_pre_serialization_order": True,
        "saved_report_matches_post_serialization_order": False,
        "saved_report_characters": len(saved_report),
        "regenerated_report_characters": len(serialized_order_report),
        "first_difference_character": first_difference,
        "failure_record": str(FAILURE_DOC),
        "failure_record_sha256": base._file_hash(base.REPO_ROOT / FAILURE_DOC),
    }


def _protocol_record_v2() -> dict[str, object]:
    record = _BASE_PROTOCOL_RECORD()
    failed_protocol = base._read_json(base.REPO_ROOT / FAILED_OUTPUT / "protocol.json")
    inherited = {field: record[field] for field in SCIENTIFIC_FIELDS}
    failed_inherited = {field: failed_protocol[field] for field in SCIENTIFIC_FIELDS}
    if inherited != failed_inherited:
        raise ValueError("PI-002 scientific protocol differs from frozen PI-001")
    record["status"] = "frozen_administrative_rerun_after_verifier_defect"
    record["failed_predecessor"] = _failed_predecessor()
    record["method_delta"] = {
        "scientific_changes": [],
        "administrative": {
            "experiment_id": EXPERIMENT_ID,
            "schema_version": SCHEMA_VERSION,
        },
        "rendering": "rescue report records sorted lexicographically before write and verify",
        "epistemic_role": (
            "administrative full rerun frozen after PI-001 outcomes; matching results "
            "are one experiment and must not be double-counted"
        ),
        "inherited_scientific_protocol_sha256": base.sha256(
            base._canonical_json_bytes(failed_inherited)
        ).hexdigest(),
    }
    return record


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


def _protect_failed_output(output: Path) -> None:
    resolved = output.resolve()
    failed = (base.REPO_ROOT / FAILED_OUTPUT).resolve()
    if resolved == failed or failed in resolved.parents or resolved in failed.parents:
        raise ValueError("PI-002 output cannot overwrite or contain preserved PI-001")


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    _protect_failed_output(output)
    with _configured():
        return base.prepare(output)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _protect_failed_output(output)
    with _configured():
        return base.run(output)


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
    semantic: bool = False,
) -> dict[str, object]:
    _protect_failed_output(output)
    with _configured():
        return base.verify(output, require_results=require_results, semantic=semantic)


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _protect_failed_output(output)
    with _configured():
        return base.analyze(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="PI-002 administrative proposal-injection rerun")
    parser.add_argument(
        "command",
        choices=("prepare", "run", "verify", "verify-semantic", "analyze"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
        print(f"prepare: {result['status']}")
    elif args.command == "run":
        result = run(args.output)
        decision = cast(dict[str, object], result["decision"])
        print(f"run: {decision['classification']}")
    elif args.command == "analyze":
        result = analyze(args.output)
        decision = cast(dict[str, object], result["decision"])
        print(f"analyze: {decision['classification']}")
    else:
        result = verify(
            args.output,
            require_results=True,
            semantic=args.command == "verify-semantic",
        )
        print(f"verify: {result['outcomes']}")


if __name__ == "__main__":
    main()
