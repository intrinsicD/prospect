"""Fail-closed tests for the backend-neutral MM-008 v2.1 lifecycle controls."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest

from bench.multimodal_mechanism_diagnostics import control_v2 as control


def _source_payload(relative: str) -> bytes:
    imports = {
        "bench/multimodal_mechanism_diagnostics/method.py": (
            "from bench.multimodal_resolution_diagnostics import method\n"
        ),
        "bench/multimodal_mechanism_diagnostics/method_v2.py": (
            "from bench.multimodal_mechanism_diagnostics import method\n"
        ),
        "bench/multimodal_mechanism_diagnostics/synthetic_v2.py": (
            "from bench.multimodal_mechanism_diagnostics import calibration_v2, method, method_v2\n"
        ),
        "bench/multimodal_resolution_diagnostics/method.py": (
            "from bench.multimodal_horizon_diagnostics import method\n"
            "from bench.multimodal_preflight import dataset\n"
            "from bench.multimodal_warp_diagnostics import method as warp\n"
        ),
        "bench/multimodal_warp_diagnostics/method.py": (
            "from bench.multimodal_horizon_diagnostics import method\n"
            "from bench.multimodal_preflight import dataset\n"
        ),
        "bench/multimodal_horizon_diagnostics/method.py": (
            "from bench.multimodal_preflight import dataset\n"
        ),
    }
    return imports.get(relative, f'"""fixture for {relative}"""\n').encode()


def _fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    for relative in control.ARCHIVE_MEMBER_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_source_payload(relative))
        path.chmod(0o644)
    return root


def _module_name(relative: str) -> str:
    name = relative[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _freeze_fixture(
    tmp_path: Path,
) -> tuple[
    control._FreezePolicy,
    control.ScienceMaterials,
    dict[str, Any],
    control._SealedCapability,
]:
    root = _fake_repo(tmp_path)
    protocol_payload = b"fake protocol\n"
    protocol_sha = sha256(protocol_payload).hexdigest()
    retired_protocol_sha = sha256(b"retired fake protocol").hexdigest()
    (root / control.PROTOCOL_PATH).write_bytes(protocol_payload)

    def receipt(protocol: str, nonce: str, created: str) -> bytes:
        return control.canonical_json_bytes(
            {
                "created_at_utc": created,
                "nonce_hex": nonce,
                "protocol_sha256": protocol,
                "reviewer_id": control.NONCE_REVIEWER,
                "schema_version": control.NONCE_SCHEMA,
            }
        )

    fresh_bytes = receipt(protocol_sha, "1" * 64, "2026-07-16T00:00:00Z")
    retired_bytes = receipt(retired_protocol_sha, "2" * 64, "2026-07-15T00:00:00Z")
    (root / control.FRESH_RECEIPT_PATH).write_bytes(fresh_bytes)
    (root / control.RETIRED_RECEIPT_PATH).write_bytes(retired_bytes)

    parent_pins: list[control.ParentPin] = []
    for index, (name, _, mode) in enumerate(control._PARENT_SPECS):
        relative = f"fake-parent/{name}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"parent-{index}\n".encode()
        path.write_bytes(payload)
        path.chmod(mode)
        parent_pins.append(control.ParentPin(relative, sha256(payload).hexdigest(), mode))

    (root / "state").mkdir()
    archive = control._build_source_archive(root, "state/source-archive.json")
    (root / "formal").mkdir()
    prefix = root / ".venv"
    (prefix / "bin").mkdir(parents=True)
    real_python = root / "python-real"
    real_python.write_bytes(b"fake CPython binary")
    (prefix / "bin/python").symlink_to(real_python)
    numpy_origin = prefix / "lib/python/site-packages/numpy/__init__.py"
    numpy_origin.parent.mkdir(parents=True)
    numpy_origin.write_bytes(b"fake numpy")
    executable = str((prefix / "bin/python").absolute())
    environment = (
        ("LANG", "C.UTF-8"),
        ("LC_ALL", "C.UTF-8"),
        ("MKL_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("PYTHONHASHSEED", "0"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
    )
    argv = (executable, "-m", "fixture.runner", "formal-run")
    runtime_policy = control._RuntimePolicy(
        executable, str(prefix.absolute()), str(root.absolute()), argv, environment
    )
    origins: list[control.ModuleOrigin] = []
    by_name = {_module_name(path): path for path in control.SCIENCE_SOURCE_PATHS}
    for name in control.RUNTIME_MODULE_NAMES:
        if name == "numpy":
            origin_path = numpy_origin
        else:
            origin_path = root / by_name[name]
        origins.append(
            control.ModuleOrigin(
                name,
                str(origin_path.absolute()),
                sha256(origin_path.read_bytes()).hexdigest(),
            )
        )
    runtime = control.RuntimeSnapshot(
        python_executable=executable,
        python_prefix=str(prefix.absolute()),
        python_real_executable=str(real_python.absolute()),
        python_binary_sha256=sha256(real_python.read_bytes()).hexdigest(),
        implementation="CPython",
        python_version="3.12.0",
        python_build=("main", "fixture"),
        numpy_version="2.4.6",
        numpy_origin=str(numpy_origin.absolute()),
        numpy_origin_sha256=sha256(numpy_origin.read_bytes()).hexdigest(),
        numpy_tree_sha256="a" * 64,
        numpy_build_sha256="b" * 64,
        pip_freeze=("numpy==2.4.6", "pytest==9.1.1"),
        module_origins=tuple(origins),
        sys_flags=tuple((name, 0) for name in control.SYS_FLAG_NAMES),
        cwd=str(root.absolute()),
        argv=argv,
        environment=environment,
    )
    usage = control.UsageSnapshot(tuple((name, 0) for name in control.NO_USE_COUNTER_NAMES))
    state: dict[str, Any] = {"runtime": runtime, "usage": usage}
    policy = control._FreezePolicy(
        repo_root=root,
        archive_path=archive.archive_path,
        formal_start_path="formal/formal-start.json",
        terminal_receipt_path="formal/terminal.json",
        protocol_sha256=protocol_sha,
        fresh_receipt_sha256=sha256(fresh_bytes).hexdigest(),
        retired_protocol_sha256=retired_protocol_sha,
        retired_receipt_sha256=sha256(retired_bytes).hexdigest(),
        parent_pins=tuple(parent_pins),
        runtime=runtime_policy,
    )
    materials = control.ScienceMaterials(b'{"method":"fixture"}', b"candidate-order", b'{"synthetic":"fixture"}')
    capability = control._build_freeze_capability(
        policy,
        materials,
        lambda: cast(control.RuntimeSnapshot, state["runtime"]),
        lambda: cast(control.UsageSnapshot, state["usage"]),
    )
    return policy, materials, state, capability


def _archive_value(archive: control.SourceArchive) -> dict[str, object]:
    return cast(dict[str, object], json.loads(archive.canonical_bytes))


def _replace_archive(root: Path, archive: control.SourceArchive, value: object) -> None:
    path = root / archive.archive_path
    path.chmod(0o644)
    path.write_bytes(control.canonical_json_bytes(value))
    path.chmod(0o444)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{"a":Infinity}\n',
        b'\xef\xbb\xbf{"a":1}\n',
        b'{"a":"\xc3\xa9"}\n',
        b'{"a":1}\r\n',
        b'{"a":1}',
        b'{ "a":1}\n',
        b'{"b":2,"a":1}\n',
        b'{"a":1}\n\n',
        b'{"a":1e999}\n',
    ],
)
def test_canonical_json_rejects_malformed_or_noncanonical(payload: bytes) -> None:
    with pytest.raises(control.ControlV2Error):
        control.parse_canonical_json_bytes(payload)


def test_canonical_json_rejects_nonfinite_and_non_json_values() -> None:
    with pytest.raises(control.ControlV2Error):
        control.canonical_json_bytes({"value": float("nan")})
    with pytest.raises(control.ControlV2Error):
        control.canonical_json_bytes({"value": (1, 2)})


def test_archive_build_validate_and_exact_closure(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    assert control.validate_science_import_closure(root) == control.SCIENCE_SOURCE_PATHS
    archive = control._build_source_archive(root, "source-archive.json")
    assert archive.entries[0].path == control.ARCHIVE_MEMBER_PATHS[0]
    assert tuple(entry.path for entry in archive.entries) == control.ARCHIVE_MEMBER_PATHS
    assert len(archive.entries) == 22
    assert (root / "source-archive.json").stat().st_mode & 0o777 == 0o444
    assert control._validate_source_archive(root, "source-archive.json") == archive
    with pytest.raises(control.TerminalRunError):
        control._build_source_archive(root, "source-archive.json")


@pytest.mark.parametrize("mutation", ["missing", "extra", "reorder", "duplicate", "bool_mode"])
def test_archive_rejects_membership_and_typed_integer_tamper(
    tmp_path: Path, mutation: str
) -> None:
    root = _fake_repo(tmp_path)
    archive = control._build_source_archive(root, "source-archive.json")
    value = _archive_value(archive)
    files = value["files"]
    assert isinstance(files, list)
    if mutation == "missing":
        files.pop()
    elif mutation == "extra":
        files.append(dict(files[-1], path="unexpected.py"))
    elif mutation == "reorder":
        files[0], files[1] = files[1], files[0]
    elif mutation == "duplicate":
        files[1] = dict(files[0])
    else:
        files[0] = dict(files[0], mode=True)
    _replace_archive(root, archive, value)
    with pytest.raises(control.ControlV2Error):
        control._validate_source_archive(root, archive.archive_path)


@pytest.mark.parametrize("field", ["bytes_b64", "size", "sha256", "mode"])
def test_archive_rejects_record_tamper(tmp_path: Path, field: str) -> None:
    root = _fake_repo(tmp_path)
    archive = control._build_source_archive(root, "source-archive.json")
    value = _archive_value(archive)
    files = value["files"]
    assert isinstance(files, list) and isinstance(files[0], dict)
    if field == "bytes_b64":
        files[0][field] = f"{files[0][field]} "
    elif field == "size":
        files[0][field] = int(files[0][field]) + 1
    elif field == "sha256":
        files[0][field] = "0" * 64
    else:
        files[0][field] = 0o600
    _replace_archive(root, archive, value)
    with pytest.raises(control.ControlV2Error):
        control._validate_source_archive(root, archive.archive_path)


def test_archive_rejects_live_byte_and_mode_drift(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    archive = control._build_source_archive(root, "source-archive.json")
    target = root / control.MM008_TEST_PATHS[0]
    target.write_bytes(target.read_bytes() + b"# drift\n")
    with pytest.raises(control.ControlV2Error):
        control._validate_source_archive(root, archive.archive_path)

    root = _fake_repo(tmp_path / "second")
    archive = control._build_source_archive(root, "source-archive.json")
    (root / control.MM008_TEST_PATHS[0]).chmod(0o600)
    with pytest.raises(control.ControlV2Error):
        control._validate_source_archive(root, archive.archive_path)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_archive_rejects_nonregular_or_multilink_members(
    tmp_path: Path, kind: str
) -> None:
    root = _fake_repo(tmp_path)
    target = root / control.MM008_TEST_PATHS[0]
    target.unlink()
    if kind == "symlink":
        target.symlink_to(root / control.MM008_TEST_PATHS[1])
    elif kind == "hardlink":
        os.link(root / control.MM008_TEST_PATHS[1], target)
    else:
        os.mkfifo(target)
    with pytest.raises(control.ControlV2Error):
        control._build_source_archive(root, "source-archive.json")


def test_archive_rejects_symlink_ancestor_and_path_escape(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    real_bench = root / "real-bench"
    (root / "bench").rename(real_bench)
    (root / "bench").symlink_to(real_bench, target_is_directory=True)
    with pytest.raises(control.ControlV2Error):
        control._build_source_archive(root, "source-archive.json")
    for path in ("", "/absolute", "../escape", "a/../escape", "a\\b", "a\x00b"):
        with pytest.raises(control.ControlV2Error):
            control._validate_relative_path(path)


def test_ast_closure_rejects_missing_and_extra_local_import(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    method_path = root / "bench/multimodal_mechanism_diagnostics/method.py"
    method_path.write_text("\n")
    with pytest.raises(control.ControlV2Error):
        control.validate_science_import_closure(root)

    root = _fake_repo(tmp_path / "extra")
    extra = root / "bench/unexpected.py"
    extra.write_text("\n")
    mechanism = root / "bench/multimodal_mechanism_diagnostics/control_v2.py"
    mechanism.write_text("import bench.unexpected\n")
    with pytest.raises(control.ControlV2Error):
        control.validate_science_import_closure(root)


def test_live_exact_thirteen_closure_is_stop_ship_due_package_initializers() -> None:
    with pytest.raises(control.ControlV2Error):
        control.validate_science_import_closure(control.REPO_ROOT)
    with pytest.raises(control.ControlV2Error, match=control.LIFECYCLE_ADMISSION_STATUS):
        control.build_source_archive()
    with pytest.raises(control.ControlV2Error, match=control.LIFECYCLE_ADMISSION_STATUS):
        control.validate_source_archive()


def test_exact_coverage_cardinalities_hashes_and_receipt() -> None:
    assert tuple(
        map(
            len,
            (
                control.SCENARIO_KEYS,
                control.ARM_BANK_KEYS,
                control.ROW_ARM_KEYS,
                control.ORACLE_KEYS,
                control.MUTATION_KEYS,
                control.TRANSPOSE_PAIR_ROW_KEYS,
                control.TRANSPOSE_MODE_KEYS,
            ),
        )
    ) == (8, 34, 204, 12, 4, 36, 108)
    results = tuple(
        control.CoverageResult(key, f"{index:064x}", f"{index + 1:064x}")
        for index, key in enumerate(control.EXPECTED_COVERAGE_KEYS)
    )
    receipt = control.CoverageReceipt("f" * 64, results)
    assert control.parse_coverage_receipt_structure(receipt.canonical_bytes) == receipt
    assert tuple(result.key for result in receipt.results) == control.EXPECTED_COVERAGE_KEYS
    assert {key.parts[-1] for key in control.TRANSPOSE_MODE_KEYS} == set(control.MODE_ORDER)


@pytest.mark.parametrize("mutation", ["remove", "duplicate", "add", "reorder"])
def test_coverage_rejects_key_level_drift(mutation: str) -> None:
    values = list(control.EXPECTED_COVERAGE_KEYS)
    if mutation == "remove":
        values.pop()
    elif mutation == "duplicate":
        values[-1] = values[0]
    elif mutation == "add":
        values.append(control.CoverageKey("unexpected", ("extra",)))
    else:
        values[0], values[1] = values[1], values[0]
    with pytest.raises(control.ControlV2Error):
        control.validate_coverage_keys(values)


def test_coverage_receipt_rejects_noncanonical_hash_and_bool_int() -> None:
    results = tuple(
        control.CoverageResult(key, "d" * 64, "e" * 64)
        for key in control.EXPECTED_COVERAGE_KEYS
    )
    receipt = control.CoverageReceipt("f" * 64, results)
    value = receipt.as_dict()
    value["coverage_schema_sha256"] = "0" * 64
    with pytest.raises(control.ControlV2Error):
        control.parse_coverage_receipt_structure(control.canonical_json_bytes(value))
    with pytest.raises(control.ControlV2Error):
        control._require_int(True, "fixture")
    with pytest.raises(control.ControlV2Error):
        replace(receipt, results=receipt.results[:-1])


def test_fake_freeze_recomputes_all_bound_layers(tmp_path: Path) -> None:
    policy, materials, state, capability = _freeze_fixture(tmp_path)
    record = capability.freeze_record
    assert record.protocol.sha256 == policy.protocol_sha256
    assert record.archive.path == policy.archive_path
    assert len(record.parent_files) == 8
    assert record.fresh_receipt.status == "fresh_authorized"
    assert record.retired_receipt.status == "retired_unused"
    assert record.science == control.ScienceHashes.from_materials(materials)
    assert record.runtime == state["runtime"]
    assert control._revalidate_freeze_capability_test_only(capability) == record
    with pytest.raises(control.ControlV2Error, match=control.LIFECYCLE_ADMISSION_STATUS):
        control.build_freeze_capability(
            materials,
            lambda: cast(control.RuntimeSnapshot, state["runtime"]),
            lambda: cast(control.UsageSnapshot, state["usage"]),
        )
    with pytest.raises(control.ControlV2Error, match=control.LIFECYCLE_ADMISSION_STATUS):
        control.revalidate_freeze_capability(capability)
    with pytest.raises(control.ControlV2Error, match=control.LIFECYCLE_ADMISSION_STATUS):
        control.write_formal_start(capability)
    with pytest.raises(control.ControlV2Error):
        control._SealedCapability(
            policy,
            materials,
            lambda: cast(control.RuntimeSnapshot, state["runtime"]),
            lambda: cast(control.UsageSnapshot, state["usage"]),
            record,
            token=object(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "python_executable",
        "python_binary_hash",
        "cwd",
        "argv",
        "environment",
        "module_origin",
        "module_hash",
        "module_order",
        "sys_flags",
        "pip_freeze",
        "numpy_origin",
    ],
)
def test_freeze_rejects_wrong_runtime_fields(tmp_path: Path, mutation: str) -> None:
    policy, materials, state, capability = _freeze_fixture(tmp_path)
    runtime = cast(control.RuntimeSnapshot, state["runtime"])
    if mutation == "python_executable":
        changed = replace(runtime, python_executable=str(policy.repo_root / "wrong-python"))
    elif mutation == "python_binary_hash":
        changed = replace(runtime, python_binary_sha256="0" * 64)
    elif mutation == "cwd":
        changed = replace(runtime, cwd=str(policy.repo_root / "wrong"))
    elif mutation == "argv":
        changed = replace(runtime, argv=(*runtime.argv, "--alternate"))
    elif mutation == "environment":
        changed = replace(runtime, environment=runtime.environment[:-1])
    elif mutation in {"module_origin", "module_hash"}:
        origins = list(runtime.module_origins)
        first = origins[0]
        origins[0] = replace(
            first,
            origin=(str(policy.repo_root / "wrong.py") if mutation == "module_origin" else first.origin),
            sha256=("0" * 64 if mutation == "module_hash" else first.sha256),
        )
        changed = replace(runtime, module_origins=tuple(origins))
    elif mutation == "module_order":
        changed = replace(runtime, module_origins=tuple(reversed(runtime.module_origins)))
    elif mutation == "sys_flags":
        flags = list(runtime.sys_flags)
        flags[0] = (flags[0][0], cast(Any, True))
        changed = replace(runtime, sys_flags=tuple(flags))
    elif mutation == "pip_freeze":
        changed = replace(runtime, pip_freeze=tuple(reversed(runtime.pip_freeze)))
    else:
        changed = replace(runtime, numpy_origin=str(policy.repo_root / "outside-numpy.py"))
    state["runtime"] = changed
    with pytest.raises(control.ControlV2Error):
        control._revalidate_freeze_capability_test_only(capability)
    with pytest.raises(control.ControlV2Error):
        control._build_freeze_capability(
            policy,
            materials,
            lambda: cast(control.RuntimeSnapshot, state["runtime"]),
            lambda: cast(control.UsageSnapshot, state["usage"]),
        )


def test_freeze_rejects_nonzero_or_bool_no_use_counter(tmp_path: Path) -> None:
    _, _, state, capability = _freeze_fixture(tmp_path)
    usage = cast(control.UsageSnapshot, state["usage"])
    counters = list(usage.counters)
    counters[0] = (counters[0][0], 1)
    state["usage"] = control.UsageSnapshot(tuple(counters))
    with pytest.raises(control.ControlV2Error):
        control._revalidate_freeze_capability_test_only(capability)
    counters[0] = (counters[0][0], cast(Any, True))
    state["usage"] = control.UsageSnapshot(tuple(counters))
    with pytest.raises(control.ControlV2Error):
        control._revalidate_freeze_capability_test_only(capability)


def test_source_drift_before_marker_prevents_marker(tmp_path: Path) -> None:
    policy, _, _, capability = _freeze_fixture(tmp_path)
    target = policy.repo_root / control.MM008_TEST_PATHS[0]
    target.write_bytes(target.read_bytes() + b"drift")
    with pytest.raises(control.ControlV2Error):
        control._write_formal_start_test_only(capability)
    assert not (policy.repo_root / policy.formal_start_path).exists()


def test_marker_race_is_exclusive_and_presence_is_terminal(tmp_path: Path) -> None:
    policy, _, _, capability = _freeze_fixture(tmp_path)

    def attempt() -> str:
        try:
            control._write_formal_start_test_only(capability)
        except control.TerminalRunError:
            return "terminal"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ["created", "terminal"]
    marker = policy.repo_root / policy.formal_start_path
    assert marker.stat().st_mode & 0o777 == 0o444
    assert not (policy.repo_root / policy.terminal_receipt_path).exists()
    with pytest.raises(control.TerminalRunError):
        control._write_formal_start_test_only(capability)


def test_failed_receipt_and_marker_never_allow_rerun(tmp_path: Path) -> None:
    policy, _, _, capability = _freeze_fixture(tmp_path)
    control._write_formal_start_test_only(capability)
    terminal = control._write_failure_receipt_test_only(
        capability, failure_code="fixture_failure", diagnostic_payload=b"diagnostic"
    )
    assert terminal.mode == 0o444
    parsed = control._validate_terminal_receipt(capability, check_live=False)
    assert parsed["status"] == "failed"
    assert parsed["diagnostic_b64"] == "ZGlhZ25vc3RpYw=="
    with pytest.raises(control.TerminalRunError):
        control._write_failure_receipt_test_only(
            capability, failure_code="second_failure", diagnostic_payload=b"second"
        )
    with pytest.raises(control.TerminalRunError):
        control._write_formal_start_test_only(capability)
    assert (policy.repo_root / policy.formal_start_path).exists()


def test_preexisting_partial_terminal_blocks_marker(tmp_path: Path) -> None:
    policy, _, _, capability = _freeze_fixture(tmp_path)
    terminal = policy.repo_root / policy.terminal_receipt_path
    terminal.write_bytes(b"partial")
    with pytest.raises(control.TerminalRunError):
        control._write_formal_start_test_only(capability)
    assert not (policy.repo_root / policy.formal_start_path).exists()


def test_completed_terminal_receipt_has_pre_post_revalidation(tmp_path: Path) -> None:
    _, _, _, capability = _freeze_fixture(tmp_path)
    control._write_formal_start_test_only(capability)
    receipt = control.TerminalReceipt(
        "completed", sha256(b"result").hexdigest(), None, None
    )
    control._write_terminal_receipt_test_only(capability, receipt)
    parsed = control._validate_terminal_receipt_test_only(capability)
    assert parsed["status"] == "completed"


def test_post_marker_source_drift_blocks_completion_but_allows_failure_receipt(
    tmp_path: Path,
) -> None:
    policy, _, _, capability = _freeze_fixture(tmp_path)
    control._write_formal_start_test_only(capability)
    target = policy.repo_root / control.MM008_TEST_PATHS[0]
    target.write_bytes(target.read_bytes() + b"post marker drift")
    with pytest.raises(control.ControlV2Error):
        control._write_terminal_receipt_test_only(
            capability,
            control.TerminalReceipt(
                "completed", sha256(b"result").hexdigest(), None, None
            ),
        )
    control._write_failure_receipt_test_only(
        capability, failure_code="source_drift", diagnostic_payload=b"drift"
    )
    assert control._validate_terminal_receipt(capability, check_live=False)["status"] == "failed"
    with pytest.raises(control.TerminalRunError):
        control._write_formal_start_test_only(capability)


def test_post_marker_receipt_comes_only_from_marker_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy, _, _, capability = _freeze_fixture(tmp_path)
    expected = capability.freeze_record.fresh_receipt.receipt_dict
    control._write_formal_start_test_only(capability)
    external = policy.repo_root / control.FRESH_RECEIPT_PATH
    external.chmod(0o644)
    external.write_bytes(b"externally destroyed after marker")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("post-marker code touched external receipt parsing")

    monkeypatch.setattr(control, "_binding_from_archive", forbidden)
    assert control._challenge_receipt_from_formal_start_test_only(capability) == expected


def test_drift_during_marker_write_leaves_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy, _, _, capability = _freeze_fixture(tmp_path)
    original = control._write_exclusive_readonly

    def write_then_drift(root: Path, relative: str, payload: bytes) -> control.AnchoredFile:
        result = original(root, relative, payload)
        if relative == policy.formal_start_path:
            target = root / control.MM008_TEST_PATHS[0]
            target.write_bytes(target.read_bytes() + b"raced drift")
        return result

    monkeypatch.setattr(control, "_write_exclusive_readonly", write_then_drift)
    with pytest.raises(control.TerminalRunError):
        control._write_formal_start_test_only(capability)
    assert (policy.repo_root / policy.formal_start_path).exists()
    with pytest.raises(control.ControlV2Error):
        control._write_formal_start_test_only(capability)
