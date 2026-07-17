"""PI-003 namespace and semantic-canonicalization delta over PI-002."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from bench.proposal_injection import experiment as base
from bench.proposal_injection_v2 import experiment as v2

SCHEMA_VERSION = "proposal-injection-v3"
EXPERIMENT_ID = "PI-003"
DEFAULT_OUTPUT = Path("bench/proposal_injection_v3/results/PI-003")
PROTOCOL_DOC = Path("docs/research/2026-07-14-proposal-injection-pi003-protocol.md")
FAILURE_DOC = Path("docs/research/2026-07-14-pi002-semantic-verifier-failure.md")
FAILED_OUTPUT = Path("bench/proposal_injection_v2/results/PI-002")

FAILED_HASHES = {
    "protocol.json": "ed50d151a8bb6d7c485528bc53eaa64b8e26cb8a111003a4ef6019c1ca233da3",
    "input-manifest.json": "b692c700443958c5037711cf99601bdf92f491dcc5585ba151661117bba8a0e9",
    "PI-002-results.json": "6afcfe0da6d954b1cf1399bf66b99c5c084701a73188e1f25f705cfca4f0b4ad",
    "PI-002-runs.csv": "bb003b16ce91b918561381982ebd4e0c6eedae5c05ef3de59318a7bf8fd63c5e",
    "PI-002-report.md": "c462cc64e7bc02812be352a2f3d76bab237078ca54a88d7d39710eb2cdd0965f",
    "inputs/BC-001-b1_r1_d8.npz": "9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948",
    "artifact-manifest.json": "8a08f97da5c100191fcac1348d06fd1dbcbb5a615bb89672052583a71408fd3d",
}

SCIENTIFIC_FIELDS = v2.SCIENTIFIC_FIELDS
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *base.SOURCE_FILES,
            *v2.SOURCE_FILES,
            Path("bench/proposal_injection_v3/__init__.py"),
            Path("bench/proposal_injection_v3/__main__.py"),
            Path("bench/proposal_injection_v3/experiment.py"),
            Path("tests/test_proposal_injection_v3.py"),
            PROTOCOL_DOC,
            FAILURE_DOC,
        )
    )
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


def _canonical_json_value(value: object) -> object:
    """Normalize only the exact finite JSON representation used by artifacts."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _report_text_v3(results: dict[str, Any]) -> str:
    revised = dict(results)
    decision = dict(cast(dict[str, Any], results["decision"]))
    rescues = cast(dict[str, Any], decision["rescues"])
    decision["rescues"] = {name: rescues[name] for name in sorted(rescues)}
    revised["decision"] = decision
    return _BASE_REPORT_TEXT(revised).replace(
        "# PI-001 proposal-injection result",
        "# PI-003 proposal-injection result",
        1,
    )


def _failed_predecessor() -> dict[str, object]:
    root = base.REPO_ROOT / FAILED_OUTPUT
    actual = {name: base._file_hash(root / name) for name in FAILED_HASHES}
    if actual != FAILED_HASHES:
        raise ValueError("preserved PI-002 failed artifacts have drifted")
    failed_protocol = base._read_json(root / "protocol.json")
    current_sources = {
        path: base._file_hash(base.REPO_ROOT / path)
        for path in cast(dict[str, str], failed_protocol["source_sha256"])
    }
    if current_sources != failed_protocol["source_sha256"]:
        raise ValueError("PI-002 frozen source snapshot has drifted")
    failed_manifest = base._read_json(root / "artifact-manifest.json")
    expected_manifest = {
        name: digest for name, digest in FAILED_HASHES.items() if name != "artifact-manifest.json"
    }
    if failed_manifest.get("artifacts") != expected_manifest:
        raise ValueError("PI-002 artifact manifest does not match the pinned package")

    failed_results = base._read_json(root / "PI-002-results.json")
    provider_row = next(
        row
        for row in cast(list[dict[str, Any]], failed_results["rows"])
        if row["arm"] == "privileged_injection" and row["seed"] == 0
    )
    provider = cast(dict[str, Any], provider_row["provider"])
    first_call = cast(dict[str, Any], cast(list[dict[str, Any]], provider["calls"])[0])
    reference_scores = first_call["reference_exact_scores"]
    output_scores = first_call["output_exact_scores"]
    if not isinstance(reference_scores, list) or not isinstance(output_scores, list):
        raise ValueError("PI-002 saved provider scores are not JSON arrays")
    regenerated_shape = {
        **first_call,
        "reference_exact_scores": tuple(reference_scores),
        "output_exact_scores": tuple(output_scores),
    }
    if first_call == regenerated_shape:
        raise ValueError("PI-002 semantic defect is not the recorded tuple/list mismatch")
    if first_call != _canonical_json_value(regenerated_shape):
        raise ValueError("PI-002 tuple/list mismatch is not JSON-canonical-equivalent")
    return {
        "experiment_id": "PI-002",
        "path": str(FAILED_OUTPUT),
        "terminal_failure": "semantic verifier compared regenerated tuples with JSON-decoded lists",
        "artifact_sha256": actual,
        "artifact_manifest_entries_verified": True,
        "source_snapshot_matches_protocol": True,
        "saved_provider_scores_are_lists": True,
        "regenerated_provider_scores_are_tuples": True,
        "raw_container_equality": False,
        "canonical_json_equality": True,
        "full_regeneration_diagnostic": {
            "rows_raw_equal": False,
            "rows_canonical_equal": True,
            "parity_canonical_equal": True,
            "decision_canonical_equal": True,
            "executed_arms_canonical_equal": True,
        },
        "failure_record": str(FAILURE_DOC),
        "failure_record_sha256": base._file_hash(base.REPO_ROOT / FAILURE_DOC),
    }


def _protocol_record_v3() -> dict[str, object]:
    record = _BASE_PROTOCOL_RECORD()
    failed_protocol = base._read_json(base.REPO_ROOT / FAILED_OUTPUT / "protocol.json")
    inherited = {field: record[field] for field in SCIENTIFIC_FIELDS}
    failed_inherited = {field: failed_protocol[field] for field in SCIENTIFIC_FIELDS}
    if inherited != failed_inherited:
        raise ValueError("PI-003 scientific protocol differs from frozen PI-002")
    record["status"] = "frozen_administrative_rerun_after_semantic_verifier_defect"
    record["failed_predecessor"] = _failed_predecessor()
    record["method_delta"] = {
        "scientific_changes": [],
        "administrative": {
            "experiment_id": EXPERIMENT_ID,
            "schema_version": SCHEMA_VERSION,
        },
        "rendering_inherited": "rescue report records sorted lexicographically",
        "semantic_verification": (
            "regenerated fields normalized through exact finite sorted-key JSON before equality"
        ),
        "epistemic_role": (
            "administrative full rerun frozen after PI-001/PI-002 outcomes; all three "
            "packages are one experiment and must not be double-counted"
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
        "protocol_record": _protocol_record_v3,
        "_report_text": _report_text_v3,
    }
    original = {name: getattr(base, name) for name in patches}
    try:
        for name, value in patches.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def _protect_failed_outputs(output: Path) -> None:
    resolved = output.resolve()
    for failed_path in (v2.FAILED_OUTPUT, FAILED_OUTPUT):
        failed = (base.REPO_ROOT / failed_path).resolve()
        if resolved == failed or failed in resolved.parents or resolved in failed.parents:
            raise ValueError("PI-003 output cannot overwrite a preserved predecessor")


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    _protect_failed_outputs(output)
    with _configured():
        return base.prepare(output)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _protect_failed_outputs(output)
    with _configured():
        return base.run(output)


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
    semantic: bool = False,
) -> dict[str, object]:
    _protect_failed_outputs(output)
    with _configured():
        result = base.verify(output, require_results=require_results, semantic=False)
        if semantic:
            if result["outcomes"] == "prepared_only":
                raise ValueError("complete PI-003 results are required for semantic verification")
            saved = base._read_json(output / f"{EXPERIMENT_ID}-results.json")
            regenerated = cast(dict[str, object], _canonical_json_value(base._execute(output / base.INPUT_COPY)))
            for field in ("rows", "parity", "decision", "executed_arms"):
                if saved[field] != regenerated[field]:
                    raise ValueError(f"PI-003 canonical semantic regeneration differs in {field}")
            result = {**result, "outcomes": "verified_semantic_results"}
        return result


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _protect_failed_outputs(output)
    with _configured():
        return base.analyze(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="PI-003 administrative proposal-injection rerun")
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
