"""Captured, isolated execution for independent WM-001 auditors.

The public executor in this module is deliberately independent of any
particular auditor CLI.  It captures an auditor and its support files, starts
the interpreter with ``-I -S -B``, restores only explicitly declared
third-party import roots, and requires a canonical JSON audit report whose
``passed`` value agrees with the process return code.

The runtime manifest contains no outcome artifact path, temporary path, or
descriptor number.  Its bytes can therefore be sealed before a formal run and
supplied again during adjudication.  A separate canonical invocation manifest
binds each run's working directory and arguments.  Temporary descriptor
details are delivered through a private control environment which the
bootstrap removes before running the auditor.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import stat
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, NoReturn, cast

from .assurance import ASSURANCE

RUNTIME_MANIFEST_SCHEMA = "prospect.wm001.audit-runtime-manifest.v1"
INVOCATION_MANIFEST_SCHEMA = "prospect.wm001.audit-invocation-manifest.v1"
SourceMode = Literal["path", "descriptor"]
CAPTURED_ARGUMENT_PREFIX = "@captured/"

_MAX_SOURCE_BYTES = 64 << 20
_MAX_SUPPORT_BYTES = 64 << 20
_MAX_MANIFEST_BYTES = 4 << 20
_MAX_INVOCATION_BYTES = 1 << 20
_MAX_REPORT_BYTES = 64 << 20
_MAX_STDERR_BYTES = 16 << 20
_AUDIT_TIMEOUT_SECONDS = 600
_BOOTSTRAP_FAILURE = 125
_CONTROL_PREFIX = "PROSPECT_AUDIT_RUNNER_"
_PACKAGE_ROOT_DOMAIN = b"prospect.wm001.package-root.v2\0"
_STDLIB_ROOT_DOMAIN = b"prospect.wm001.standard-library.v2\0"
_MANIFEST_FD_ENV = f"{_CONTROL_PREFIX}MANIFEST_FD"
_INVOCATION_FD_ENV = f"{_CONTROL_PREFIX}INVOCATION_FD"
_BOOTSTRAP_FD_ENV = f"{_CONTROL_PREFIX}BOOTSTRAP_FD"
_CAPTURE_ROOT_ENV = f"{_CONTROL_PREFIX}CAPTURE_ROOT"
_SOURCE_PATH_ENV = f"{_CONTROL_PREFIX}SOURCE_PATH"
_SOURCE_FD_ENV = f"{_CONTROL_PREFIX}SOURCE_FD"

# These values can affect numerical runtime behaviour without enabling Python's
# ambient module-discovery machinery.  Callers must still opt in by supplying
# each value explicitly; nothing is inherited implicitly.
SAFE_RUNTIME_ENVIRONMENT_KEYS = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "LAZY_LEGACY_OP",
        "LANG",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYGAME_HIDE_SUPPORT_PROMPT",
        "ROCR_VISIBLE_DEVICES",
        "SDL_AUDIODRIVER",
        "TZ",
    }
)


class AuditRunnerError(RuntimeError):
    """The isolated auditor could not be executed with intact custody."""


class AuditExecutionFailure(AuditRunnerError):
    """Authenticated, bounded evidence from one failed isolated execution."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        command: tuple[str, ...],
        returncode: int | None,
        stdout: bytes,
        stderr: bytes,
        runtime_manifest: bytes,
        invocation_manifest: bytes,
        bootstrap_sha256: str,
        auditor_source_sha256: str,
        support_files: tuple[CapturedFileIdentity, ...],
        source_mode: SourceMode,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.runtime_manifest = runtime_manifest
        self.runtime_manifest_sha256 = hashlib.sha256(runtime_manifest).hexdigest()
        self.invocation_manifest = invocation_manifest
        self.invocation_manifest_sha256 = hashlib.sha256(invocation_manifest).hexdigest()
        self.bootstrap_sha256 = bootstrap_sha256
        self.auditor_source_sha256 = auditor_source_sha256
        self.support_files = support_files
        self.source_mode = source_mode


@dataclass(frozen=True)
class CapturedFileIdentity:
    """Stable identity for one captured source or support file."""

    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class AuditExecution:
    """A completed audit whose canonical report agrees with its return code."""

    command: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    report: Mapping[str, object]
    runtime_manifest: bytes
    runtime_manifest_sha256: str
    invocation_manifest: bytes
    invocation_manifest_sha256: str
    bootstrap_sha256: str
    auditor_source_sha256: str
    support_files: tuple[CapturedFileIdentity, ...]
    source_mode: SourceMode


@dataclass(frozen=True)
class AuditConformance:
    """Byte-identical private-path and inherited-descriptor audit executions."""

    path_execution: AuditExecution
    descriptor_execution: AuditExecution
    path_executions: tuple[AuditExecution, ...]
    descriptor_executions: tuple[AuditExecution, ...]
    repeat_count: int
    report_sha256: str


@dataclass(frozen=True)
class _FileCapture:
    path: Path
    payload: bytes
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


# The child bootstrap is a self-contained standard-library program because it
# is launched with -S.  Keep execution-only FDs and private paths out of the
# canonical manifest so its digest is stable across runs.
_BOOTSTRAP_SOURCE = rb"""from __future__ import annotations

import hashlib
import json
import os
import runpy
import stat
import sys
import sysconfig
import traceback
from pathlib import Path, PurePosixPath

SCHEMA = "prospect.wm001.audit-runtime-manifest.v1"
INVOCATION_SCHEMA = "prospect.wm001.audit-invocation-manifest.v1"
CONTROL_PREFIX = "PROSPECT_AUDIT_RUNNER_"
MANIFEST_FD_ENV = CONTROL_PREFIX + "MANIFEST_FD"
INVOCATION_FD_ENV = CONTROL_PREFIX + "INVOCATION_FD"
BOOTSTRAP_FD_ENV = CONTROL_PREFIX + "BOOTSTRAP_FD"
CAPTURE_ROOT_ENV = CONTROL_PREFIX + "CAPTURE_ROOT"
SOURCE_PATH_ENV = CONTROL_PREFIX + "SOURCE_PATH"
SOURCE_FD_ENV = CONTROL_PREFIX + "SOURCE_FD"
FAILURE = 125
MAX_MANIFEST = 4 << 20
MAX_INVOCATION = 1 << 20
MAX_SOURCE = 64 << 20
MAX_SUPPORT = 64 << 20
MAX_REPORT = 64 << 20
MAX_STDERR = 16 << 20
TIMEOUT_SECONDS = 600
CAPTURED_ARGUMENT_PREFIX = "@captured/"
PACKAGE_DOMAIN = b"prospect.wm001.package-root.v2\0"
STDLIB_DOMAIN = b"prospect.wm001.standard-library.v2\0"
ASSURANCE = {
    "trust_model_id": "prospect.wm001.trust-model.v1",
    "tamper_resistant": False,
    "external_attestation": False,
    "exclusive_path_use_required": True,
}


class BootstrapError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def pairs(pairs_value):
    result = {}
    for key, value in pairs_value:
        if key in result:
            raise BootstrapError("runtime manifest contains a duplicate object key")
        result[key] = value
    return result


def reject_constant(value):
    raise BootstrapError("runtime manifest contains a non-finite JSON value: " + value)


def parse_manifest(payload):
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError("runtime manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or payload != canonical(value):
        raise BootstrapError("runtime manifest is not canonical JSON followed by one newline")
    return value


def expect_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise BootstrapError(label + " has an unexpected field set")


def fd_path(descriptor):
    for prefix in ("/proc/self/fd/", "/dev/fd/"):
        candidate = prefix + str(descriptor)
        if os.path.exists(candidate):
            return candidate
    raise BootstrapError("the platform has no inherited-descriptor path")


def stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_descriptor(descriptor, limit, label):
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BootstrapError(label + " descriptor is not a regular file")
        if before.st_size > limit:
            raise BootstrapError(label + " exceeds its byte limit")
        payload = os.pread(descriptor, before.st_size + 1, 0)
        after = os.fstat(descriptor)
    except OSError as error:
        raise BootstrapError(label + " descriptor cannot be read") from error
    if len(payload) != before.st_size or stat_identity(before) != stat_identity(after):
        raise BootstrapError(label + " descriptor changed while it was read")
    return payload, before


def reject_symlink_components(path, label):
    if not path.is_absolute():
        raise BootstrapError(label + " must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise BootstrapError(label + " has a symbolic-link path component")
        except FileNotFoundError as error:
            raise BootstrapError(label + " does not exist") from error


def canonical_directory(raw, label):
    if not isinstance(raw, str) or not raw:
        raise BootstrapError(label + " is invalid")
    path = Path(raw)
    reject_symlink_components(path, label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BootstrapError(label + " cannot be resolved") from error
    if resolved != path or not path.is_dir():
        raise BootstrapError(label + " must be one canonical directory")
    return path


def root_inventory(root, domain, standard_library, label):
    digest = hashlib.sha256(domain)
    file_count = 0
    directory_count = 0
    total_bytes = 0
    inventory_files = []
    inventory_directories = []
    for directory, directory_names, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        retained_directories = []
        for name in directory_names:
            candidate = Path(directory) / name
            try:
                mode = os.lstat(candidate).st_mode
            except OSError as error:
                raise BootstrapError(
                    label + " inventory directory cannot be inspected"
                ) from error
            if stat.S_ISLNK(mode):
                raise BootstrapError(label + " inventory contains a symbolic link")
            if not stat.S_ISDIR(mode):
                raise BootstrapError(label + " inventory contains a special directory")
            if not (standard_library and name == "site-packages"):
                retained_directories.append(name)
                inventory_directories.append(candidate)
        directory_names[:] = retained_directories
        for filename in filenames:
            path = Path(directory) / filename
            try:
                mode = os.lstat(path).st_mode
            except OSError as error:
                raise BootstrapError(label + " inventory entry cannot be inspected") from error
            if stat.S_ISLNK(mode):
                raise BootstrapError(label + " inventory contains a symbolic link")
            if not stat.S_ISREG(mode):
                raise BootstrapError(label + " inventory contains a special file")
            inventory_files.append(path)
    entries = (
        [("directory", path) for path in inventory_directories]
        + [("file", path) for path in inventory_files]
    )
    for kind, path in sorted(
        entries,
        key=lambda item: item[1].relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if kind == "directory":
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0directory\0")
            directory_count += 1
            continue
        payload, metadata = read_regular(
            path,
            32 << 30,
            label + " inventory entry",
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0file\0")
        digest.update(metadata.st_size.to_bytes(8, "big", signed=False))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        file_count += 1
        total_bytes += metadata.st_size
    if file_count == 0:
        raise BootstrapError(label + " inventory is empty")
    return {
        "semantics_id": (
            "prospect.wm001.standard-library.v2"
            if standard_library
            else "prospect.wm001.package-root.v2"
        ),
        "path": str(root),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def validate_inventory_row(value, label):
    expect_keys(
        value,
        {
            "path",
            "semantics_id",
            "file_count",
            "directory_count",
            "total_bytes",
            "inventory_sha256",
        },
        label,
    )
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or value["semantics_id"]
        not in {
            "prospect.wm001.package-root.v2",
            "prospect.wm001.standard-library.v2",
        }
        or type(value["file_count"]) is not int
        or value["file_count"] < 1
        or type(value["directory_count"]) is not int
        or value["directory_count"] < 0
        or type(value["total_bytes"]) is not int
        or value["total_bytes"] < 1
        or not isinstance(value["inventory_sha256"], str)
        or len(value["inventory_sha256"]) != 64
    ):
        raise BootstrapError(label + " is invalid")


def read_regular(path, limit, label):
    reject_symlink_components(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BootstrapError(label + " cannot be opened") from error
    try:
        return read_descriptor(descriptor, limit, label)
    finally:
        os.close(descriptor)


def verify_identity(payload, expected, label):
    if not isinstance(expected, dict) or set(expected) not in (
        {"path", "bytes", "sha256"},
        {"mode", "path", "bytes", "sha256"},
    ):
        raise BootstrapError(label + " identity has an unexpected field set")
    if (
        not isinstance(expected["path"], str)
        or type(expected["bytes"]) is not int
        or expected["bytes"] < 0
        or not isinstance(expected["sha256"], str)
        or len(expected["sha256"]) != 64
        or len(payload) != expected["bytes"]
        or hashlib.sha256(payload).hexdigest() != expected["sha256"]
    ):
        raise BootstrapError(label + " bytes do not match the runtime manifest")


def valid_relative_path(raw):
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return False
    path = PurePosixPath(raw)
    return not path.is_absolute() and "." not in path.parts and ".." not in path.parts


def exact_flags():
    return {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }


def validate_manifest(value):
    expect_keys(
        value,
        {
            "schema",
            "assurance",
            "bootstrap_sha256",
            "python",
            "required_flags",
            "source",
            "support_files",
            "closure_import_roots",
            "standard_library",
            "environment",
            "limits",
        },
        "runtime manifest",
    )
    if value["schema"] != SCHEMA:
        raise BootstrapError("runtime manifest schema is invalid")
    if value["assurance"] != ASSURANCE:
        raise BootstrapError("runtime manifest assurance boundary is invalid")
    if not isinstance(value["bootstrap_sha256"], str) or len(value["bootstrap_sha256"]) != 64:
        raise BootstrapError("runtime manifest bootstrap digest is invalid")
    expect_keys(
        value["python"],
        {"executable", "resolved_executable", "sha256", "version"},
        "runtime manifest python block",
    )
    try:
        executable_payload = Path(sys.executable).resolve().read_bytes()
    except OSError as error:
        raise BootstrapError("child interpreter bytes cannot be read") from error
    if (
        value["python"]["executable"] != sys.executable
        or value["python"]["resolved_executable"] != str(Path(sys.executable).resolve())
        or value["python"]["sha256"] != hashlib.sha256(executable_payload).hexdigest()
        or value["python"]["version"] != [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]
    ):
        raise BootstrapError("runtime manifest identifies a different interpreter")
    if value["required_flags"] != exact_flags():
        raise BootstrapError("runtime manifest does not require the exact isolated flags")
    actual_flags = {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
    }
    if actual_flags != value["required_flags"] or len(sys.argv) != 1:
        raise BootstrapError("child interpreter flags or bootstrap argv are not exact")
    expect_keys(value["source"], {"mode", "path", "bytes", "sha256"}, "runtime manifest source block")
    if value["source"]["mode"] not in {"path", "descriptor"}:
        raise BootstrapError("runtime manifest source mode is invalid")
    if not valid_relative_path(value["source"]["path"]):
        raise BootstrapError("runtime manifest source path is invalid")
    if not isinstance(value["support_files"], list):
        raise BootstrapError("runtime manifest support_files is invalid")
    support_paths = []
    for row in value["support_files"]:
        expect_keys(row, {"path", "bytes", "sha256"}, "runtime manifest support row")
        if not valid_relative_path(row["path"]):
            raise BootstrapError("runtime manifest support path is invalid")
        support_paths.append(row["path"])
    if len(support_paths) != len(set(support_paths)) or value["source"]["path"] in support_paths:
        raise BootstrapError("runtime manifest has duplicate captured paths")
    if support_paths != sorted(support_paths):
        raise BootstrapError("runtime manifest support paths are not ordered")
    if not isinstance(value["closure_import_roots"], list):
        raise BootstrapError("runtime manifest closure roots are invalid")
    root_paths = []
    for index, row in enumerate(value["closure_import_roots"]):
        validate_inventory_row(row, "runtime manifest closure root")
        root_paths.append(row["path"])
    if len(root_paths) != len(set(root_paths)):
        raise BootstrapError("runtime manifest closure roots are invalid or duplicated")
    validate_inventory_row(
        value["standard_library"],
        "runtime manifest standard library",
    )
    environment = value["environment"]
    if (
        not isinstance(environment, dict)
        or any(
            not isinstance(key, str)
            or not key
            or key.startswith("PYTHON")
            or key.startswith(CONTROL_PREFIX)
            or not isinstance(item, str)
            or "\x00" in key
            or "\x00" in item
            for key, item in environment.items()
        )
    ):
        raise BootstrapError("runtime manifest environment is invalid")
    if value["limits"] != {
        "timeout_seconds": TIMEOUT_SECONDS,
        "stdout_bytes": MAX_REPORT,
        "stderr_bytes": MAX_STDERR,
    }:
        raise BootstrapError("runtime manifest execution limits are invalid")


def captured_argument_relative(raw):
    if not isinstance(raw, str) or not raw.startswith("@captured"):
        return None
    if not raw.startswith(CAPTURED_ARGUMENT_PREFIX):
        raise BootstrapError("invocation contains a malformed captured-support token")
    relative = raw[len(CAPTURED_ARGUMENT_PREFIX):]
    if not valid_relative_path(relative):
        raise BootstrapError("invocation captured-support token has an unsafe path")
    return relative


def validate_invocation(value, runtime_payload, runtime):
    expect_keys(
        value,
        {
            "schema",
            "runtime_manifest_sha256",
            "working_directory",
            "auditor_argv",
        },
        "invocation manifest",
    )
    if value["schema"] != INVOCATION_SCHEMA:
        raise BootstrapError("invocation manifest schema is invalid")
    if value["runtime_manifest_sha256"] != hashlib.sha256(runtime_payload).hexdigest():
        raise BootstrapError("invocation manifest identifies a different runtime manifest")
    if (
        not isinstance(value["working_directory"], str)
        or not value["working_directory"]
        or "\x00" in value["working_directory"]
    ):
        raise BootstrapError("invocation manifest working directory is invalid")
    if not isinstance(value["auditor_argv"], list) or any(
        not isinstance(item, str) or "\x00" in item
        for item in value["auditor_argv"]
    ):
        raise BootstrapError("invocation manifest auditor argv is invalid")
    support_paths = {row["path"] for row in runtime["support_files"]}
    for argument in value["auditor_argv"]:
        relative = captured_argument_relative(argument)
        if relative is not None and relative not in support_paths:
            raise BootstrapError("invocation references an unknown captured support file")


def execute():
    try:
        manifest_fd = int(os.environ[MANIFEST_FD_ENV])
        invocation_fd = int(os.environ[INVOCATION_FD_ENV])
        bootstrap_fd = int(os.environ[BOOTSTRAP_FD_ENV])
        capture_root_raw = os.environ[CAPTURE_ROOT_ENV]
    except (KeyError, ValueError) as error:
        raise BootstrapError("private bootstrap controls are incomplete") from error

    manifest_payload, _ = read_descriptor(manifest_fd, MAX_MANIFEST, "runtime manifest")
    manifest = parse_manifest(manifest_payload)
    validate_manifest(manifest)
    invocation_payload, _ = read_descriptor(
        invocation_fd,
        MAX_INVOCATION,
        "invocation manifest",
    )
    invocation = parse_manifest(invocation_payload)
    validate_invocation(invocation, manifest_payload, manifest)

    bootstrap_payload, bootstrap_stat = read_descriptor(bootstrap_fd, MAX_SOURCE, "bootstrap")
    if hashlib.sha256(bootstrap_payload).hexdigest() != manifest["bootstrap_sha256"]:
        raise BootstrapError("bootstrap bytes do not match the runtime manifest")

    source_mode = manifest["source"]["mode"]
    control_keys = {
        MANIFEST_FD_ENV,
        INVOCATION_FD_ENV,
        BOOTSTRAP_FD_ENV,
        CAPTURE_ROOT_ENV,
    }
    if source_mode == "descriptor":
        control_keys.add(SOURCE_FD_ENV)
    else:
        control_keys.add(SOURCE_PATH_ENV)
    expected_environment = dict(manifest["environment"])
    if set(os.environ) != set(expected_environment) | control_keys:
        raise BootstrapError("child environment contains undeclared variables")
    for key, value in expected_environment.items():
        if os.environ.get(key) != value:
            raise BootstrapError("child environment differs from the runtime manifest")

    capture_root = canonical_directory(capture_root_raw, "private capture root")
    private_source = capture_root / manifest["source"]["path"]
    private_payload, private_stat = read_regular(private_source, MAX_SOURCE, "private auditor source")
    verify_identity(private_payload, manifest["source"], "auditor source")

    source_descriptor = None
    if source_mode == "descriptor":
        try:
            source_descriptor = int(os.environ[SOURCE_FD_ENV])
        except (KeyError, ValueError) as error:
            raise BootstrapError("source descriptor control is invalid") from error
        source_payload, source_stat = read_descriptor(source_descriptor, MAX_SOURCE, "auditor source")
        verify_identity(source_payload, manifest["source"], "auditor source")
        if (source_stat.st_dev, source_stat.st_ino) != (private_stat.st_dev, private_stat.st_ino):
            raise BootstrapError("source descriptor is not the private captured source")
        execution_path = fd_path(source_descriptor)
    else:
        execution_path = os.environ.get(SOURCE_PATH_ENV)
        if execution_path != str(private_source):
            raise BootstrapError("source path control is not the private captured source")

    supports = []
    support_argument_paths = {}
    for row in manifest["support_files"]:
        support_path = capture_root.joinpath(*PurePosixPath(row["path"]).parts)
        payload, support_stat = read_regular(support_path, MAX_SUPPORT, "captured support file")
        verify_identity(payload, row, "captured support file")
        supports.append((support_path, payload, support_stat, row))
        support_argument_paths[row["path"]] = str(support_path)

    working_directory = canonical_directory(invocation["working_directory"], "auditor working directory")
    stdlib_row = manifest["standard_library"]
    stdlib_root = canonical_directory(
        stdlib_row["path"],
        "standard-library root",
    )
    if stdlib_root != Path(sysconfig.get_path("stdlib")):
        raise BootstrapError("runtime manifest identifies a different standard library")
    stdlib_inventory = root_inventory(
        stdlib_root,
        STDLIB_DOMAIN,
        True,
        "standard-library root",
    )
    if stdlib_inventory != stdlib_row:
        raise BootstrapError("standard-library inventory differs before auditor import")
    root_rows = manifest["closure_import_roots"]
    roots = []
    root_inventories = []
    for row in root_rows:
        root = canonical_directory(row["path"], "closure import root")
        inventory = root_inventory(
            root,
            PACKAGE_DOMAIN,
            False,
            "closure import root",
        )
        if inventory != row:
            raise BootstrapError("closure import-root inventory differs before auditor import")
        roots.append(root)
        root_inventories.append(inventory)
    baseline = list(sys.path)
    sys.path[:] = [str(root) for root in roots] + baseline
    os.chdir(working_directory)
    resolved_arguments = []
    for argument in invocation["auditor_argv"]:
        relative = captured_argument_relative(argument)
        resolved_arguments.append(
            argument if relative is None else support_argument_paths[relative]
        )
    sys.argv = [execution_path, *resolved_arguments]
    for key in tuple(os.environ):
        if key.startswith(CONTROL_PREFIX):
            del os.environ[key]

    exit_code = 0
    try:
        runpy.run_path(execution_path, run_name="__main__")
    except SystemExit as error:
        if error.code is None:
            exit_code = 0
        elif isinstance(error.code, int):
            exit_code = error.code
        else:
            print(error.code, file=sys.stderr)
            exit_code = 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1

    # The auditor may be hostile or simply broken.  Reopen every captured byte
    # identity before allowing its exit status to escape the bootstrap.
    after_manifest, _ = read_descriptor(manifest_fd, MAX_MANIFEST, "runtime manifest")
    if after_manifest != manifest_payload:
        raise BootstrapError("runtime manifest changed during execution")
    after_invocation, _ = read_descriptor(
        invocation_fd,
        MAX_INVOCATION,
        "invocation manifest",
    )
    if after_invocation != invocation_payload:
        raise BootstrapError("invocation manifest changed during execution")
    if root_inventory(
        stdlib_root,
        STDLIB_DOMAIN,
        True,
        "standard-library root",
    ) != stdlib_inventory:
        raise BootstrapError("standard-library inventory changed during auditor execution")
    for root, expected_inventory in zip(roots, root_inventories):
        if root_inventory(
            root,
            PACKAGE_DOMAIN,
            False,
            "closure import root",
        ) != expected_inventory:
            raise BootstrapError("closure import-root inventory changed during auditor execution")
    after_bootstrap, after_bootstrap_stat = read_descriptor(bootstrap_fd, MAX_SOURCE, "bootstrap")
    if after_bootstrap != bootstrap_payload or stat_identity(after_bootstrap_stat) != stat_identity(bootstrap_stat):
        raise BootstrapError("bootstrap changed during execution")
    after_private, after_private_stat = read_regular(private_source, MAX_SOURCE, "private auditor source")
    if after_private != private_payload or stat_identity(after_private_stat) != stat_identity(private_stat):
        raise BootstrapError("private auditor source changed during execution")
    if source_descriptor is not None:
        after_source, after_source_stat = read_descriptor(source_descriptor, MAX_SOURCE, "auditor source")
        if after_source != private_payload or stat_identity(after_source_stat) != stat_identity(private_stat):
            raise BootstrapError("auditor source descriptor changed during execution")
    for support_path, payload, support_stat, _ in supports:
        after_payload, after_stat = read_regular(support_path, MAX_SUPPORT, "captured support file")
        if after_payload != payload or stat_identity(after_stat) != stat_identity(support_stat):
            raise BootstrapError("captured support file changed during execution")
    return exit_code


def main():
    try:
        return execute()
    except BootstrapError as error:
        print("audit-runner bootstrap: " + str(error), file=sys.stderr)
        return FAILURE
    except BaseException:
        traceback.print_exc()
        return FAILURE


raise SystemExit(main())
"""

BOOTSTRAP_SHA256 = hashlib.sha256(_BOOTSTRAP_SOURCE).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _parse_canonical_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AuditRunnerError(f"{label} contains duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise AuditRunnerError(f"{label} contains non-finite JSON value {value}")

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditRunnerError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise AuditRunnerError(f"{label} must contain one JSON object")
    if payload != _canonical_json_bytes(decoded):
        raise AuditRunnerError(f"{label} is not canonical JSON followed by one newline")
    return cast(dict[str, object], decoded)


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise AuditRunnerError(f"{label} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except OSError as error:
            raise AuditRunnerError(f"{label} cannot be resolved: {error}") from error
        if stat.S_ISLNK(mode):
            raise AuditRunnerError(f"{label} has a symbolic-link path component")


def _canonical_directory(path: Path, *, label: str) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    _reject_symlink_components(candidate, label=label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise AuditRunnerError(f"{label} cannot be resolved: {error}") from error
    if resolved != candidate or not candidate.is_dir():
        raise AuditRunnerError(f"{label} must be one canonical directory")
    return candidate


def _root_inventory(
    root: Path,
    *,
    domain: bytes,
    standard_library: bool,
) -> dict[str, object]:
    canonical = _canonical_directory(root, label="runtime inventory root")
    digest = hashlib.sha256(domain)
    file_count = 0
    directory_count = 0
    total_bytes = 0
    inventory_files: list[Path] = []
    inventory_directories: list[Path] = []
    for directory, directory_names, filenames in os.walk(
        canonical,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        retained_directories: list[str] = []
        for name in directory_names:
            directory_path = Path(directory) / name
            try:
                mode = directory_path.lstat().st_mode
            except OSError as error:
                raise AuditRunnerError(
                    f"runtime inventory directory cannot be inspected: {directory_path.relative_to(canonical)}"
                ) from error
            if stat.S_ISLNK(mode):
                raise AuditRunnerError(
                    f"runtime inventory contains a symbolic-link directory: {directory_path.relative_to(canonical)}"
                )
            if not stat.S_ISDIR(mode):
                raise AuditRunnerError(
                    f"runtime inventory contains a special directory: {directory_path.relative_to(canonical)}"
                )
            if not (standard_library and name == "site-packages"):
                retained_directories.append(name)
                inventory_directories.append(directory_path)
        directory_names[:] = retained_directories
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(canonical).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise AuditRunnerError(f"runtime inventory contains a symbolic link: {relative}")
            if not stat.S_ISREG(mode):
                raise AuditRunnerError(f"runtime inventory contains a special file: {relative}")
            inventory_files.append(path)
    entries = [
        *(("directory", path) for path in inventory_directories),
        *(("file", path) for path in inventory_files),
    ]
    for kind, path in sorted(
        entries,
        key=lambda item: item[1].relative_to(canonical).as_posix(),
    ):
        relative = path.relative_to(canonical).as_posix()
        if kind == "directory":
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0directory\0")
            directory_count += 1
            continue
        capture = _capture_regular_file(
            path,
            limit=32 << 30,
            label=f"runtime inventory entry {relative!r}",
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0file\0")
        digest.update(capture.size.to_bytes(8, "big", signed=False))
        digest.update(b"\0")
        digest.update(capture.payload)
        digest.update(b"\0")
        file_count += 1
        total_bytes += capture.size
    if file_count == 0:
        raise AuditRunnerError("runtime inventory root is empty")
    return {
        "semantics_id": (
            "prospect.wm001.standard-library.v2" if standard_library else "prospect.wm001.package-root.v2"
        ),
        "path": str(canonical),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def _capture_regular_file(path: Path, *, limit: int, label: str) -> _FileCapture:
    candidate = path if path.is_absolute() else Path.cwd() / path
    _reject_symlink_components(candidate, label=label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise AuditRunnerError(f"{label} cannot be resolved: {error}") from error
    if resolved != candidate:
        raise AuditRunnerError(f"{label} must have one canonical path")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise AuditRunnerError(f"{label} cannot be opened: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuditRunnerError(f"{label} must be a regular non-symbolic-link file")
        if before.st_size > limit:
            raise AuditRunnerError(f"{label} exceeds its {limit}-byte limit")
        payload = os.pread(descriptor, before.st_size + 1, 0)
        after = os.fstat(descriptor)
    except OSError as error:
        raise AuditRunnerError(f"{label} cannot be read: {error}") from error
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size or _stat_identity(before) != _stat_identity(after):
        raise AuditRunnerError(f"{label} changed while it was read")
    return _FileCapture(
        path=candidate,
        payload=payload,
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        modified_ns=before.st_mtime_ns,
        changed_ns=before.st_ctime_ns,
    )


def _recheck_capture(capture: _FileCapture, *, limit: int, label: str) -> None:
    current = _capture_regular_file(capture.path, limit=limit, label=label)
    if (
        current.payload != capture.payload
        or current.device != capture.device
        or current.inode != capture.inode
        or current.size != capture.size
        or current.modified_ns != capture.modified_ns
        or current.changed_ns != capture.changed_ns
    ):
        raise AuditRunnerError(f"{label} changed during isolated execution")


def _validate_relative_capture_path(raw: str, *, label: str) -> str:
    if not raw or "\\" in raw:
        raise AuditRunnerError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise AuditRunnerError(f"{label} has an unsafe path")
    return path.as_posix()


def captured_support_argument(relative_path: str) -> str:
    """Return the stable invocation token for one manifest-declared support."""

    relative = _validate_relative_capture_path(
        relative_path,
        label="captured-support argument",
    )
    return f"{CAPTURED_ARGUMENT_PREFIX}{relative}"


def _validate_invocation_arguments(
    arguments: Sequence[str],
    *,
    runtime: Mapping[str, object],
) -> list[str]:
    result = list(arguments)
    support_rows = runtime.get("support_files")
    if not isinstance(support_rows, list):
        raise AuditRunnerError("runtime manifest support-file block is invalid")
    support_paths = {
        row.get("path") for row in support_rows if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    for argument in result:
        if not isinstance(argument, str) or "\0" in argument:
            raise AuditRunnerError("auditor arguments must be NUL-free strings")
        if argument.startswith("@captured"):
            if not argument.startswith(CAPTURED_ARGUMENT_PREFIX):
                raise AuditRunnerError("auditor argument contains a malformed captured-support token")
            relative = _validate_relative_capture_path(
                argument.removeprefix(CAPTURED_ARGUMENT_PREFIX),
                label="captured-support argument",
            )
            if relative not in support_paths:
                raise AuditRunnerError(f"auditor argument references unknown captured support {relative!r}")
    return result


def _validated_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    result = dict(environment or {})
    for key, value in result.items():
        if key not in SAFE_RUNTIME_ENVIRONMENT_KEYS:
            raise AuditRunnerError(f"runtime environment key {key!r} is not explicitly safe")
        if key.startswith("PYTHON") or key.startswith(_CONTROL_PREFIX):
            raise AuditRunnerError(f"runtime environment key {key!r} is forbidden")
        if not isinstance(value, str) or "\0" in key or "\0" in value:
            raise AuditRunnerError("runtime environment keys and values must be NUL-free strings")
    # A fixed locale prevents CPython's pre-bootstrap locale coercion from
    # injecting an undeclared LC_CTYPE value into an otherwise empty env.
    result.setdefault("LC_ALL", "C.UTF-8")
    return dict(sorted(result.items()))


def _limit_output_files() -> None:
    """Apply an OS-enforced per-file ceiling before the isolated exec."""

    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (_MAX_REPORT_BYTES, _MAX_REPORT_BYTES),
    )


def _prepare_inputs(
    auditor_source: Path,
    *,
    support_files: Mapping[str, Path] | None,
    closure_import_roots: Sequence[Path],
    environment: Mapping[str, str] | None,
) -> tuple[
    _FileCapture,
    tuple[tuple[str, _FileCapture], ...],
    tuple[Path, ...],
    dict[str, str],
]:
    source = _capture_regular_file(
        auditor_source,
        limit=_MAX_SOURCE_BYTES,
        label="auditor source",
    )
    supports: list[tuple[str, _FileCapture]] = []
    seen = {source.path.name}
    for raw_relative, path in (support_files or {}).items():
        relative = _validate_relative_capture_path(
            raw_relative,
            label="support-file destination",
        )
        if relative in seen:
            raise AuditRunnerError(f"duplicate captured path {relative!r}")
        seen.add(relative)
        supports.append(
            (
                relative,
                _capture_regular_file(
                    path,
                    limit=_MAX_SUPPORT_BYTES,
                    label=f"support file {relative!r}",
                ),
            )
        )
    supports.sort(key=lambda item: item[0])
    captured_paths = sorted(seen)
    if any(
        left != right
        and (
            PurePosixPath(left).is_relative_to(PurePosixPath(right))
            or PurePosixPath(right).is_relative_to(PurePosixPath(left))
        )
        for index, left in enumerate(captured_paths)
        for right in captured_paths[index + 1 :]
    ):
        raise AuditRunnerError("captured file paths must not contain one another")
    roots = tuple(_canonical_directory(path, label="closure import root") for path in closure_import_roots)
    if len(roots) != len(set(roots)):
        raise AuditRunnerError("closure import roots must be unique")
    return source, tuple(supports), roots, _validated_environment(environment)


def _manifest_from_inputs(
    source: _FileCapture,
    *,
    source_mode: SourceMode,
    supports: tuple[tuple[str, _FileCapture], ...],
    roots: tuple[Path, ...],
    environment: Mapping[str, str],
) -> bytes:
    if source_mode not in {"path", "descriptor"}:
        raise AuditRunnerError(f"unsupported auditor source mode: {source_mode!r}")
    source_name = _validate_relative_capture_path(
        source.path.name,
        label="auditor source name",
    )
    value: dict[str, object] = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "assurance": dict(ASSURANCE),
        "bootstrap_sha256": BOOTSTRAP_SHA256,
        "python": {
            "executable": sys.executable,
            "resolved_executable": str(Path(sys.executable).resolve()),
            "sha256": _sha256(Path(sys.executable).resolve().read_bytes()),
            "version": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
        },
        "required_flags": {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
            "safe_path": True,
        },
        "source": {
            "mode": source_mode,
            "path": source_name,
            "bytes": len(source.payload),
            "sha256": _sha256(source.payload),
        },
        "support_files": [
            {
                "path": relative,
                "bytes": len(capture.payload),
                "sha256": _sha256(capture.payload),
            }
            for relative, capture in supports
        ],
        "closure_import_roots": [
            _root_inventory(
                root,
                domain=_PACKAGE_ROOT_DOMAIN,
                standard_library=False,
            )
            for root in roots
        ],
        "standard_library": _root_inventory(
            Path(sysconfig.get_path("stdlib")),
            domain=_STDLIB_ROOT_DOMAIN,
            standard_library=True,
        ),
        "environment": dict(environment),
        "limits": {
            "timeout_seconds": _AUDIT_TIMEOUT_SECONDS,
            "stdout_bytes": _MAX_REPORT_BYTES,
            "stderr_bytes": _MAX_STDERR_BYTES,
        },
    }
    payload = _canonical_json_bytes(value)
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise AuditRunnerError("canonical runtime manifest exceeds its byte limit")
    return payload


def build_runtime_manifest(
    auditor_source: Path,
    *,
    support_files: Mapping[str, Path] | None = None,
    closure_import_roots: Sequence[Path] = (),
    source_mode: SourceMode = "descriptor",
    environment: Mapping[str, str] | None = None,
) -> bytes:
    """Build stable canonical bytes suitable for a pre-outcome binding.

    Files are read only to derive their immutable identities.  The executor
    captures and checks them again when the manifest is used.
    """

    source, supports, roots, safe_environment = _prepare_inputs(
        auditor_source,
        support_files=support_files,
        closure_import_roots=closure_import_roots,
        environment=environment,
    )
    return _manifest_from_inputs(
        source,
        source_mode=source_mode,
        supports=supports,
        roots=roots,
        environment=safe_environment,
    )


def build_invocation_manifest(
    *,
    runtime_manifest: bytes,
    auditor_arguments: Sequence[str] = (),
    working_directory: Path | None = None,
) -> bytes:
    """Bind one run's target and argv without changing sealed runtime identity."""

    runtime = _parse_canonical_json_object(
        runtime_manifest,
        label="runtime manifest",
    )
    if runtime.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise AuditRunnerError("runtime manifest has the wrong schema")
    arguments = _validate_invocation_arguments(
        auditor_arguments,
        runtime=runtime,
    )
    cwd = _canonical_directory(
        working_directory or Path.cwd(),
        label="auditor working directory",
    )
    payload = _canonical_json_bytes(
        {
            "schema": INVOCATION_MANIFEST_SCHEMA,
            "runtime_manifest_sha256": _sha256(runtime_manifest),
            "working_directory": str(cwd),
            "auditor_argv": arguments,
        }
    )
    if len(payload) > _MAX_INVOCATION_BYTES:
        raise AuditRunnerError("canonical invocation manifest exceeds its byte limit")
    return payload


def bootstrap_source_sha256() -> str:
    """Return the stable identity of the descriptor-executed bootstrap."""

    return BOOTSTRAP_SHA256


def bootstrap_source_bytes() -> bytes:
    """Return the exact bootstrap bytes for binding and artifact preservation."""

    return _BOOTSTRAP_SOURCE


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AuditRunnerError(f"could not write private capture {path.name!r}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _open_read_descriptor(path: Path, *, label: str) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise AuditRunnerError(f"{label} cannot be opened: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise AuditRunnerError(f"{label} is not a regular file")
    return descriptor


def _descriptor_path(descriptor: int) -> str:
    for prefix in ("/proc/self/fd", "/dev/fd"):
        candidate = Path(prefix) / str(descriptor)
        if candidate.exists():
            return str(candidate)
    raise AuditRunnerError("platform cannot execute a descriptor-bound bootstrap")


def _read_descriptor_payload(
    descriptor: int,
    *,
    limit: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuditRunnerError(f"{label} descriptor is not a regular file")
        if before.st_size > limit:
            raise AuditRunnerError(f"{label} exceeds its {limit}-byte limit")
        payload = os.pread(descriptor, before.st_size + 1, 0)
        after = os.fstat(descriptor)
    except OSError as error:
        raise AuditRunnerError(f"{label} descriptor cannot be read: {error}") from error
    if len(payload) != before.st_size or _stat_identity(before) != _stat_identity(after):
        raise AuditRunnerError(f"{label} descriptor changed while it was read")
    return payload, before


def _recheck_descriptor(
    descriptor: int,
    *,
    expected_payload: bytes,
    expected_stat: os.stat_result,
    limit: int,
    label: str,
) -> None:
    payload, metadata = _read_descriptor_payload(
        descriptor,
        limit=limit,
        label=label,
    )
    if payload != expected_payload or _stat_identity(metadata) != _stat_identity(expected_stat):
        raise AuditRunnerError(f"{label} changed during isolated execution")


def _parse_report(stdout: bytes, *, returncode: int) -> dict[str, object]:
    if len(stdout) > _MAX_REPORT_BYTES:
        raise AuditRunnerError("auditor report exceeds its byte limit")
    report = _parse_canonical_json_object(stdout, label="auditor report")
    passed = report.get("passed")
    if type(passed) is not bool:
        raise AuditRunnerError("auditor report must contain one boolean 'passed' field")
    expected_returncode = 0 if passed else 1
    if returncode != expected_returncode:
        raise AuditRunnerError(
            "auditor return code is inconsistent with its canonical report "
            f"(returncode={returncode}, passed={passed!r})"
        )
    return report


def run_captured_auditor(
    auditor_source: Path,
    *,
    auditor_arguments: Sequence[str] = (),
    support_files: Mapping[str, Path] | None = None,
    closure_import_roots: Sequence[Path] = (),
    source_mode: SourceMode = "descriptor",
    working_directory: Path | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_manifest: bytes | None = None,
    invocation_manifest: bytes | None = None,
    timeout: float | None = None,
) -> AuditExecution:
    """Run one captured auditor through the shared descriptor-bound bootstrap.

    When ``runtime_manifest`` is supplied, its bytes must exactly equal the
    manifest derived from the current source, support, closure, and runtime
    specification.  A supplied ``invocation_manifest`` must independently
    match the current argv and working directory.  This is the replay path
    used by adjudication.
    """

    if timeout not in {None, float(_AUDIT_TIMEOUT_SECONDS), _AUDIT_TIMEOUT_SECONDS}:
        raise AuditRunnerError(f"isolated auditor timeout is fixed at {_AUDIT_TIMEOUT_SECONDS} seconds")
    source, supports, roots, safe_environment = _prepare_inputs(
        auditor_source,
        support_files=support_files,
        closure_import_roots=closure_import_roots,
        environment=environment,
    )
    derived_manifest = _manifest_from_inputs(
        source,
        source_mode=source_mode,
        supports=supports,
        roots=roots,
        environment=safe_environment,
    )
    if runtime_manifest is not None:
        _parse_canonical_json_object(
            runtime_manifest,
            label="supplied runtime manifest",
        )
        if runtime_manifest != derived_manifest:
            raise AuditRunnerError("supplied runtime manifest does not exactly match the current captured runtime")
        manifest_payload = runtime_manifest
    else:
        manifest_payload = derived_manifest
    derived_invocation = build_invocation_manifest(
        runtime_manifest=manifest_payload,
        auditor_arguments=auditor_arguments,
        working_directory=working_directory,
    )
    if invocation_manifest is not None:
        _parse_canonical_json_object(
            invocation_manifest,
            label="supplied invocation manifest",
        )
        if invocation_manifest != derived_invocation:
            raise AuditRunnerError("supplied invocation manifest does not exactly match this audit invocation")
        invocation_payload = invocation_manifest
    else:
        invocation_payload = derived_invocation

    support_identities = tuple(
        CapturedFileIdentity(
            relative_path=relative,
            bytes=len(capture.payload),
            sha256=_sha256(capture.payload),
        )
        for relative, capture in supports
    )
    completed: subprocess.CompletedProcess[bytes] | None = None
    subprocess_error: BaseException | None = None
    integrity_error: AuditRunnerError | None = None
    command: tuple[str, ...] = ()
    captured_stdout = b""
    captured_stderr = b""
    with tempfile.TemporaryDirectory(prefix="prospect-audit-runner-") as temporary:
        private_root = Path(temporary).resolve(strict=True)
        capture_root = private_root / "capture"
        control_root = private_root / "control"
        capture_root.mkdir(mode=0o700)
        control_root.mkdir(mode=0o700)
        private_source = capture_root / source.path.name
        _write_exclusive(private_source, source.payload)
        for relative, capture in supports:
            destination = capture_root.joinpath(*PurePosixPath(relative).parts)
            _write_exclusive(destination, capture.payload)
        private_manifest = control_root / "runtime-manifest.json"
        private_invocation = control_root / "invocation-manifest.json"
        private_bootstrap = control_root / "bootstrap.py"
        _write_exclusive(private_manifest, manifest_payload)
        _write_exclusive(private_invocation, invocation_payload)
        _write_exclusive(private_bootstrap, _BOOTSTRAP_SOURCE)

        source_descriptor: int | None = None
        manifest_descriptor = _open_read_descriptor(
            private_manifest,
            label="private runtime manifest",
        )
        invocation_descriptor = _open_read_descriptor(
            private_invocation,
            label="private invocation manifest",
        )
        bootstrap_descriptor = _open_read_descriptor(
            private_bootstrap,
            label="private bootstrap",
        )
        if source_mode == "descriptor":
            source_descriptor = _open_read_descriptor(
                private_source,
                label="private auditor source",
            )
        stdout_path = control_root / "auditor.stdout"
        stderr_path = control_root / "auditor.stderr"

        manifest_descriptor_payload, manifest_descriptor_stat = _read_descriptor_payload(
            manifest_descriptor,
            limit=_MAX_MANIFEST_BYTES,
            label="private runtime manifest",
        )
        invocation_descriptor_payload, invocation_descriptor_stat = _read_descriptor_payload(
            invocation_descriptor,
            limit=_MAX_INVOCATION_BYTES,
            label="private invocation manifest",
        )
        bootstrap_descriptor_payload, bootstrap_descriptor_stat = _read_descriptor_payload(
            bootstrap_descriptor,
            limit=_MAX_SOURCE_BYTES,
            label="private bootstrap",
        )
        source_descriptor_payload: bytes | None = None
        source_descriptor_stat: os.stat_result | None = None
        if source_descriptor is not None:
            source_descriptor_payload, source_descriptor_stat = _read_descriptor_payload(
                source_descriptor,
                limit=_MAX_SOURCE_BYTES,
                label="private auditor source",
            )

        command = (
            sys.executable,
            "-I",
            "-S",
            "-B",
            _descriptor_path(bootstrap_descriptor),
        )
        child_environment = dict(safe_environment)
        child_environment.update(
            {
                _MANIFEST_FD_ENV: str(manifest_descriptor),
                _INVOCATION_FD_ENV: str(invocation_descriptor),
                _BOOTSTRAP_FD_ENV: str(bootstrap_descriptor),
                _CAPTURE_ROOT_ENV: str(capture_root),
            }
        )
        pass_fds = [
            manifest_descriptor,
            invocation_descriptor,
            bootstrap_descriptor,
        ]
        if source_descriptor is None:
            child_environment[_SOURCE_PATH_ENV] = str(private_source)
        else:
            child_environment[_SOURCE_FD_ENV] = str(source_descriptor)
            pass_fds.append(source_descriptor)

        try:
            completed_raw: subprocess.CompletedProcess[bytes] | None = None
            try:
                with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
                    completed_raw = subprocess.run(
                        command,
                        check=False,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        env=child_environment,
                        pass_fds=tuple(pass_fds),
                        timeout=_AUDIT_TIMEOUT_SECONDS,
                        preexec_fn=_limit_output_files,
                    )
                    stdout_stream.flush()
                    stderr_stream.flush()
                    os.fsync(stdout_stream.fileno())
                    os.fsync(stderr_stream.fileno())
            except BaseException as error:
                subprocess_error = error
            try:
                stdout_capture = _capture_regular_file(
                    stdout_path,
                    limit=_MAX_REPORT_BYTES,
                    label="isolated auditor stdout",
                )
                stderr_capture = _capture_regular_file(
                    stderr_path,
                    limit=_MAX_STDERR_BYTES,
                    label="isolated auditor stderr",
                )
                captured_stdout = stdout_capture.payload
                captured_stderr = stderr_capture.payload
            except BaseException as error:
                if subprocess_error is None:
                    subprocess_error = error
                else:
                    integrity_error = AuditRunnerError(f"isolated auditor output custody failed: {error}")
            if completed_raw is not None:
                completed = subprocess.CompletedProcess(
                    args=completed_raw.args,
                    returncode=completed_raw.returncode,
                    stdout=captured_stdout,
                    stderr=captured_stderr,
                )
            try:
                _recheck_capture(
                    source,
                    limit=_MAX_SOURCE_BYTES,
                    label="auditor source",
                )
                for relative, capture in supports:
                    _recheck_capture(
                        capture,
                        limit=_MAX_SUPPORT_BYTES,
                        label=f"support file {relative!r}",
                    )
                _recheck_descriptor(
                    manifest_descriptor,
                    expected_payload=manifest_descriptor_payload,
                    expected_stat=manifest_descriptor_stat,
                    limit=_MAX_MANIFEST_BYTES,
                    label="private runtime manifest",
                )
                _recheck_descriptor(
                    invocation_descriptor,
                    expected_payload=invocation_descriptor_payload,
                    expected_stat=invocation_descriptor_stat,
                    limit=_MAX_INVOCATION_BYTES,
                    label="private invocation manifest",
                )
                _recheck_descriptor(
                    bootstrap_descriptor,
                    expected_payload=bootstrap_descriptor_payload,
                    expected_stat=bootstrap_descriptor_stat,
                    limit=_MAX_SOURCE_BYTES,
                    label="private bootstrap",
                )
                if (
                    source_descriptor is not None
                    and source_descriptor_payload is not None
                    and source_descriptor_stat is not None
                ):
                    _recheck_descriptor(
                        source_descriptor,
                        expected_payload=source_descriptor_payload,
                        expected_stat=source_descriptor_stat,
                        limit=_MAX_SOURCE_BYTES,
                        label="private auditor source",
                    )
                private_source_capture = _capture_regular_file(
                    private_source,
                    limit=_MAX_SOURCE_BYTES,
                    label="private auditor source",
                )
                if private_source_capture.payload != source.payload:
                    raise AuditRunnerError("private auditor source changed during isolated execution")
                for relative, support in supports:
                    private_support = _capture_regular_file(
                        capture_root.joinpath(*PurePosixPath(relative).parts),
                        limit=_MAX_SUPPORT_BYTES,
                        label=f"private support file {relative!r}",
                    )
                    if private_support.payload != support.payload:
                        raise AuditRunnerError(f"private support file {relative!r} changed during isolated execution")
            except AuditRunnerError as error:
                integrity_error = error
        finally:
            if source_descriptor is not None:
                os.close(source_descriptor)
            os.close(bootstrap_descriptor)
            os.close(invocation_descriptor)
            os.close(manifest_descriptor)

    def fail(
        phase: str,
        message: str,
        *,
        cause: BaseException | None = None,
        returncode: int | None = (completed.returncode if completed is not None else None),
    ) -> NoReturn:
        failure = AuditExecutionFailure(
            message,
            phase=phase,
            command=command,
            returncode=returncode,
            stdout=captured_stdout,
            stderr=captured_stderr,
            runtime_manifest=manifest_payload,
            invocation_manifest=invocation_payload,
            bootstrap_sha256=BOOTSTRAP_SHA256,
            auditor_source_sha256=_sha256(source.payload),
            support_files=support_identities,
            source_mode=source_mode,
        )
        if cause is None:
            raise failure
        raise failure from cause

    if integrity_error is not None:
        fail(
            "integrity_check",
            f"isolated auditor integrity check failed: {integrity_error}",
            cause=integrity_error,
        )
    if subprocess_error is not None:
        if isinstance(subprocess_error, subprocess.TimeoutExpired):
            fail(
                "timeout",
                "isolated auditor execution timed out",
                cause=subprocess_error,
                returncode=None,
            )
        fail(
            "subprocess",
            f"isolated auditor execution failed: {subprocess_error}",
            cause=subprocess_error,
            returncode=None,
        )
    if completed is None:
        fail(
            "missing_process_result",
            "isolated auditor did not return a process result",
            returncode=None,
        )
    if completed.returncode == _BOOTSTRAP_FAILURE:
        diagnostic = completed.stderr[:4096].decode("utf-8", errors="replace")
        fail(
            "bootstrap_rejected",
            f"isolated bootstrap rejected execution: {diagnostic}",
        )
    try:
        report = _parse_report(
            completed.stdout,
            returncode=completed.returncode,
        )
    except AuditRunnerError as error:
        fail(
            "report_validation",
            f"isolated auditor report validation failed: {error}",
            cause=error,
        )
    return AuditExecution(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        report=MappingProxyType(report),
        runtime_manifest=manifest_payload,
        runtime_manifest_sha256=_sha256(manifest_payload),
        invocation_manifest=invocation_payload,
        invocation_manifest_sha256=_sha256(invocation_payload),
        bootstrap_sha256=BOOTSTRAP_SHA256,
        auditor_source_sha256=_sha256(source.payload),
        support_files=support_identities,
        source_mode=source_mode,
    )


def run_source_mode_conformance(
    auditor_source: Path,
    *,
    auditor_arguments: Sequence[str] = (),
    support_files: Mapping[str, Path] | None = None,
    closure_import_roots: Sequence[Path] = (),
    working_directory: Path | None = None,
    environment: Mapping[str, str] | None = None,
    repeat_count: int = 3,
    timeout: float | None = None,
) -> AuditConformance:
    """Require repeated byte identity in private-path and FD source modes."""

    if type(repeat_count) is not int or repeat_count < 1:
        raise AuditRunnerError("source-mode conformance repeat_count must be a positive integer")

    def execute(mode: SourceMode) -> AuditExecution:
        return run_captured_auditor(
            auditor_source,
            auditor_arguments=auditor_arguments,
            support_files=support_files,
            closure_import_roots=closure_import_roots,
            source_mode=mode,
            working_directory=working_directory,
            environment=environment,
            timeout=timeout,
        )

    path_executions = tuple(execute("path") for _ in range(repeat_count))
    descriptor_executions = tuple(execute("descriptor") for _ in range(repeat_count))
    path_execution = path_executions[0]
    descriptor_execution = descriptor_executions[0]
    executions = (*path_executions, *descriptor_executions)
    for execution in executions[1:]:
        if (
            execution.returncode != path_execution.returncode
            or execution.stdout != path_execution.stdout
            or execution.stderr != path_execution.stderr
            or execution.auditor_source_sha256 != path_execution.auditor_source_sha256
            or execution.support_files != path_execution.support_files
        ):
            raise AuditRunnerError(
                "repeated private-path and inherited-descriptor auditor executions are not byte-identical"
            )

    normalized_runtime: dict[str, object] | None = None
    normalized_invocation: dict[str, object] | None = None
    for index, execution in enumerate(executions):
        runtime = _parse_canonical_json_object(
            execution.runtime_manifest,
            label=f"source-mode runtime manifest {index}",
        )
        source = cast(dict[str, object], runtime["source"])
        source["mode"] = "normalized"
        invocation = _parse_canonical_json_object(
            execution.invocation_manifest,
            label=f"source-mode invocation manifest {index}",
        )
        invocation["runtime_manifest_sha256"] = "normalized"
        if normalized_runtime is None:
            normalized_runtime = runtime
            normalized_invocation = invocation
        elif runtime != normalized_runtime or invocation != normalized_invocation:
            raise AuditRunnerError(
                "private-path and inherited-descriptor execution identities differ beyond source mode"
            )
    return AuditConformance(
        path_execution=path_execution,
        descriptor_execution=descriptor_execution,
        path_executions=path_executions,
        descriptor_executions=descriptor_executions,
        repeat_count=repeat_count,
        report_sha256=_sha256(path_execution.stdout),
    )


def conformance_receipt_bytes(conformance: AuditConformance) -> bytes:
    """Serialize every repeated execution identity into one canonical receipt."""

    executions = (
        *conformance.path_executions,
        *conformance.descriptor_executions,
    )
    rows: list[dict[str, object]] = []
    for ordinal, execution in enumerate(executions, start=1):
        rows.append(
            {
                "ordinal": ordinal,
                "source_mode": execution.source_mode,
                "returncode": execution.returncode,
                "stdout": {
                    "bytes": len(execution.stdout),
                    "sha256": _sha256(execution.stdout),
                },
                "stderr": {
                    "bytes": len(execution.stderr),
                    "sha256": _sha256(execution.stderr),
                },
                "runtime_manifest": {
                    "bytes": len(execution.runtime_manifest),
                    "sha256": execution.runtime_manifest_sha256,
                },
                "invocation_manifest": {
                    "bytes": len(execution.invocation_manifest),
                    "sha256": execution.invocation_manifest_sha256,
                },
                "bootstrap_sha256": execution.bootstrap_sha256,
                "auditor_source_sha256": execution.auditor_source_sha256,
                "support_files": [
                    {
                        "path": support.relative_path,
                        "bytes": support.bytes,
                        "sha256": support.sha256,
                    }
                    for support in execution.support_files
                ],
                "auditor_report_passed": execution.report.get("passed"),
            }
        )
    value = {
        "schema": "prospect.wm001.audit-conformance-receipt.v1",
        "repeat_count": conformance.repeat_count,
        "execution_count": len(rows),
        "executions": rows,
        "report_sha256": conformance.report_sha256,
        "path_descriptor_byte_identical": True,
        "execution_conformance_passed": True,
    }
    return _canonical_json_bytes(value)


__all__ = [
    "AuditConformance",
    "AuditExecution",
    "AuditExecutionFailure",
    "AuditRunnerError",
    "BOOTSTRAP_SHA256",
    "CAPTURED_ARGUMENT_PREFIX",
    "CapturedFileIdentity",
    "INVOCATION_MANIFEST_SCHEMA",
    "RUNTIME_MANIFEST_SCHEMA",
    "SAFE_RUNTIME_ENVIRONMENT_KEYS",
    "SourceMode",
    "bootstrap_source_bytes",
    "bootstrap_source_sha256",
    "build_invocation_manifest",
    "build_runtime_manifest",
    "captured_support_argument",
    "conformance_receipt_bytes",
    "run_captured_auditor",
    "run_source_mode_conformance",
]
