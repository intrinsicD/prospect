"""Pinned runtime and numerical canary for LCV-001.

This module deliberately knows nothing about MM-001--MM-009.  It establishes
which interpreter, virtual environment, NumPy build, BLAS object, thread policy,
and SVD implementation execute the consumer-semantic verifier.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import stat
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np

SCHEMA_VERSION: Final = "lcv001-runtime-v1"
REPO_ROOT: Final = Path("/home/alex/Documents/prospect")
SHADOW_SOURCE_ROOT: Final = Path(__file__).absolute().parents[2]
LEXICAL_PYTHON: Final = REPO_ROOT / ".venv/bin/python"
VENV_PREFIX: Final = REPO_ROOT / ".venv"
BASE_PREFIX: Final = Path("/home/alex/miniconda3")
BASE_EXECUTABLE: Final = BASE_PREFIX / "bin/python3.12"
PYTHON_VERSION: Final = "3.12.9"
NUMPY_VERSION: Final = "2.4.6"
PYTHON_TERMINAL_SHA256: Final = "d9b73f600a860cefe799d7b4d21bad64719231227c2cf326ca17ed94624841af"
PYVENV_CFG_SHA256: Final = "6764aaef5351413f2ae10147dc839f27c7f1af4b9b8300338a402f3ce2f106ff"
NUMPY_INIT_SHA256: Final = "92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a"
MULTIARRAY_SHA256: Final = "2bd8ea4ada756eb29d34bcb5049df8d48710721b57c2c23cbdd87d77cdf5fa77"
UMATH_LINALG_SHA256: Final = "d826c92d6a98bda58ddb2f5a6f086d9581937538c2c2aef5b8879ed43282b995"
OPENBLAS_SHA256: Final = "05c9f9eb89ee68a4b9d673184fa91c99587e736392c0c2d49180a8aa5303d080"
OPENBLAS_CONFIG: Final = "OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64"
OPENBLAS_CORE: Final = "SkylakeX"
THREAD_ENVIRONMENT: Final = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SYMLINK_CHAIN: Final = (
    (str(LEXICAL_PYTHON), "/home/alex/miniconda3/bin/python"),
    ("/home/alex/miniconda3/bin/python", "python3.12"),
)
NUMPY_ROOT: Final = VENV_PREFIX / "lib/python3.12/site-packages/numpy"
NUMPY_INIT: Final = NUMPY_ROOT / "__init__.py"
MULTIARRAY: Final = NUMPY_ROOT / "_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so"
UMATH_LINALG: Final = NUMPY_ROOT / "linalg/_umath_linalg.cpython-312-x86_64-linux-gnu.so"
OPENBLAS: Final = VENV_PREFIX / "lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-32a4b2a6.so"
STDLIB_ROOT: Final = BASE_PREFIX / "lib/python3.12"
NUMPY_CLOSURE_COUNT: Final = 893
NUMPY_CLOSURE_BYTES: Final = 56_351_948
NUMPY_CLOSURE_SHA256: Final = "e6aa4a903960766e1b227e6717957d0a62cd8022b7953a38db17a4950b242422"
STDLIB_CLOSURE_COUNT: Final = 916
STDLIB_CLOSURE_BYTES: Final = 34_805_888
STDLIB_CLOSURE_SHA256: Final = "6275b55cbe4b2a7453542bd2d2f91662bb355c48b4792925d7daeaf0c2ef0f59"

CANARY_INPUT_SHA256: Final = "5c7a7f2b2631ab921f46b0078ec13e8e85dbf01cd7b2930224d95b1ebb8f84bd"
CANARY_U_SHA256: Final = "7cc207a6fdcb2e0d3446fa7646ce4ed4d9137ce43d91a7131024f5f576929f97"
CANARY_S_SHA256: Final = "1c81aaed50b87796b3ad37821913606ea6d2fb70a83f28b07b182a8df26d1858"
CANARY_VH_SHA256: Final = "c0aa9d926b5cdc74aaaf3df5890d8c345e2cf0f903bde39ec4b4e8c6d3a20199"
CANARY_BUNDLE_SHA256: Final = "fcce45258061121edbb0ce285c0466e13862af102a86f6a56fa6bfab8db000dd"
CANARY_MIN_GAP: Final = 0.0035610527003058223
CANARY_RELATIVE_RECONSTRUCTION: Final = 2.8895852262160284e-15


class RuntimeClosureError(ValueError):
    """The executing numerical runtime differs from the frozen closure."""


def _sha256_file(path: Path) -> str:
    before = path.lstat()
    # Package managers legitimately hard-link immutable stdlib files.  Their
    # device/inode/link-count identity is held stable across the read; LCV-owned
    # and parent artifact custody separately requires st_nlink == 1.
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeClosureError(f"runtime dependency is not a regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            != identity
        ):
            raise RuntimeClosureError(f"runtime dependency changed before open: {path}")
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeClosureError(f"runtime dependency ended early: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeClosureError(f"runtime dependency grew while read: {path}")
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise RuntimeClosureError(f"runtime dependency mutated while read: {path}")
        try:
            after_path = path.lstat()
        except FileNotFoundError as error:
            raise RuntimeClosureError(f"runtime dependency disappeared while read: {path}") from error
        if (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        ) != identity:
            raise RuntimeClosureError(f"runtime dependency path was replaced while read: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    import json

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _closure_manifest(roots: tuple[tuple[str, Path], ...], *, stdlib: bool) -> dict[str, object]:
    records: dict[str, dict[str, object]] = {}
    total = 0
    for label, root in roots:
        if root.is_symlink() or not root.is_dir():
            raise RuntimeClosureError(f"runtime closure root is not a real directory: {root}")
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            base = Path(directory)
            names[:] = sorted(
                name
                for name in names
                if name != "__pycache__" and not (stdlib and base == root and name == "site-packages")
            )
            for name in names:
                child = base / name
                if child.is_symlink() or not child.is_dir():
                    raise RuntimeClosureError(f"runtime closure has a symlink/special directory: {child}")
            for name in sorted(files):
                if name.endswith(".pyc"):
                    continue
                child = base / name
                metadata = child.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeClosureError(f"runtime closure has a symlink/special file: {child}")
                key = str(Path(label) / child.relative_to(root)) if label else str(child.relative_to(root))
                if key in records:
                    raise RuntimeClosureError(f"runtime closure repeats a normalized file key: {key}")
                records[key] = {"bytes": metadata.st_size, "sha256": _sha256_file(child)}
                total += metadata.st_size
    return {
        "bytes": total,
        "count": len(records),
        "manifest_sha256": _canonical_sha256(records),
        "records": records,
    }


def _closure_aggregate(manifest: Mapping[str, object]) -> dict[str, object]:
    records = manifest.get("records")
    if not isinstance(records, dict):
        raise RuntimeClosureError("runtime closure record map is missing")
    aggregate = {
        "bytes": manifest.get("bytes"),
        "count": manifest.get("count"),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }
    if (
        aggregate["count"] != len(records)
        or aggregate["bytes"]
        != sum(record.get("bytes", -1) for record in records.values() if isinstance(record, dict))
        or aggregate["manifest_sha256"] != _canonical_sha256(records)
    ):
        raise RuntimeClosureError("runtime closure record map does not match its aggregate")
    return aggregate


def validate_dependency_manifests(value: object) -> dict[str, dict[str, object]]:
    """Validate sealed per-file maps and return their compact commitments."""

    if not isinstance(value, dict) or set(value) != {"numpy", "stdlib"}:
        raise RuntimeClosureError("runtime dependency-manifest membership differs")
    aggregates: dict[str, dict[str, object]] = {}
    for name in ("numpy", "stdlib"):
        manifest = value[name]
        if not isinstance(manifest, dict) or set(manifest) != {
            "bytes",
            "count",
            "manifest_sha256",
            "records",
        }:
            raise RuntimeClosureError(f"{name} runtime dependency-manifest schema differs")
        records = manifest["records"]
        if not isinstance(records, dict):
            raise RuntimeClosureError(f"{name} runtime dependency records are not an object")
        for key, record in records.items():
            pure = PurePosixPath(key) if isinstance(key, str) else PurePosixPath(".")
            if (
                not isinstance(key, str)
                or not key
                or "\\" in key
                or "\x00" in key
                or pure.is_absolute()
                or any(part in {"", ".", ".."} for part in pure.parts)
                or str(pure) != key
                or not isinstance(record, dict)
                or set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int
                or record["bytes"] < 0
                or not isinstance(record["sha256"], str)
                or len(record["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in record["sha256"])
            ):
                raise RuntimeClosureError(f"{name} runtime dependency record differs: {key!r}")
        aggregates[name] = _closure_aggregate(manifest)
    expected = {
        "numpy": {
            "bytes": NUMPY_CLOSURE_BYTES,
            "count": NUMPY_CLOSURE_COUNT,
            "manifest_sha256": NUMPY_CLOSURE_SHA256,
        },
        "stdlib": {
            "bytes": STDLIB_CLOSURE_BYTES,
            "count": STDLIB_CLOSURE_COUNT,
            "manifest_sha256": STDLIB_CLOSURE_SHA256,
        },
    }
    if aggregates != expected:
        raise RuntimeClosureError("runtime dependency-manifest commitments differ")
    return aggregates


def dependency_manifests() -> dict[str, dict[str, object]]:
    """Return every pinned stdlib/NumPy file record plus its aggregate commitment."""

    numpy_manifest = _closure_manifest(
        (("numpy", NUMPY_ROOT), ("numpy.libs", OPENBLAS.parent)),
        stdlib=False,
    )
    stdlib_manifest = _closure_manifest((("", STDLIB_ROOT),), stdlib=True)
    manifests = {"numpy": numpy_manifest, "stdlib": stdlib_manifest}
    validate_dependency_manifests(manifests)
    return manifests


def dependency_closures() -> dict[str, dict[str, object]]:
    """Return compact aggregate commitments while validating every file record."""

    return validate_dependency_manifests(dependency_manifests())


def _loaded_openblas_paths() -> tuple[str, ...]:
    paths: set[str] = set()
    maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    for line in maps.splitlines():
        fields = line.split()
        if fields and "libscipy_openblas" in fields[-1]:
            paths.add(fields[-1])
    return tuple(sorted(paths))


def _openblas_observation() -> dict[str, object]:
    library = ctypes.CDLL(str(OPENBLAS))

    def integer_symbol(name: str) -> int:
        function = getattr(library, name)
        function.argtypes = []
        function.restype = ctypes.c_int
        return int(function())

    def string_symbol(name: str) -> str:
        function = getattr(library, name)
        function.argtypes = []
        function.restype = ctypes.c_char_p
        value = function()
        if not value:
            raise RuntimeClosureError(f"OpenBLAS symbol returned null: {name}")
        return cast(bytes, value).decode("ascii")

    return {
        "config": string_symbol("scipy_openblas_get_config64_"),
        "core": string_symbol("scipy_openblas_get_corename64_"),
        "parallel": integer_symbol("scipy_openblas_get_parallel64_"),
        "threads": integer_symbol("scipy_openblas_get_num_threads64_"),
    }


def _bytes_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def svd_canary() -> dict[str, object]:
    """Execute the frozen 512x256 near-degenerate SVD canary."""

    row: np.ndarray = np.arange(512, dtype=np.int64)[:, None]
    column: np.ndarray = np.arange(256, dtype=np.int64)[None, :]
    matrix = (((131 * row + 197 * column + 17 * row * column) % 1021) - 510).astype(np.float64) / 512.0
    matrix -= np.mean(matrix, axis=0, keepdims=True)
    matrix /= np.sqrt(np.mean(np.square(matrix), axis=0, keepdims=True))
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    bundle = hashlib.sha256()
    for value in (left, singular, right):
        bundle.update(np.ascontiguousarray(value).tobytes(order="C"))
    minimum_gap = float(np.min(singular[:-1] - singular[1:]))
    reconstruction = float(np.linalg.norm(matrix - (left * singular) @ right) / np.linalg.norm(matrix))
    return {
        "bundle_sha256": bundle.hexdigest(),
        "input_sha256": _bytes_sha256(matrix),
        "minimum_singular_gap": minimum_gap,
        "relative_reconstruction_error": reconstruction,
        "s_sha256": _bytes_sha256(singular),
        "schema_version": "lcv001-svd-canary-v1",
        "shape": [512, 256],
        "u_sha256": _bytes_sha256(left),
        "vh_sha256": _bytes_sha256(right),
    }


def _validate_symlink_chain() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_path, expected_target in SYMLINK_CHAIN:
        path = Path(raw_path)
        metadata = path.lstat()
        if not stat.S_ISLNK(metadata.st_mode):
            raise RuntimeClosureError(f"runtime chain member is not a symlink: {path}")
        target = os.readlink(path)
        if target != expected_target:
            raise RuntimeClosureError(f"runtime symlink target differs: {path}")
        records.append({"path": raw_path, "target": target})
    if _sha256_file(BASE_EXECUTABLE) != PYTHON_TERMINAL_SHA256:
        raise RuntimeClosureError("terminal interpreter hash differs")
    return records


def observe_and_validate() -> dict[str, object]:
    """Return a receipt only when every runtime pin and canary matches."""

    if sys.executable != str(LEXICAL_PYTHON):
        raise RuntimeClosureError(f"lexical sys.executable differs: {sys.executable}")
    if Path(sys.prefix) != BASE_PREFIX or Path(sys.base_prefix) != BASE_PREFIX:
        raise RuntimeClosureError("-S base prefix closure differs")
    if Path(sys.exec_prefix) != BASE_PREFIX or Path(sys.base_exec_prefix) != BASE_PREFIX:
        raise RuntimeClosureError("-S base exec-prefix closure differs")
    base_executable = getattr(sys, "_base_executable", None)
    if base_executable != str(BASE_EXECUTABLE):
        raise RuntimeClosureError("base executable differs")
    if platform.python_version() != PYTHON_VERSION or np.__version__ != NUMPY_VERSION:
        raise RuntimeClosureError("Python or NumPy version differs")
    expected_environment = frozen_environment()
    observed_environment = dict(os.environ)
    observed_environment.pop("LCV001_FORMAL_CHILD", None)
    if observed_environment != expected_environment:
        raise RuntimeClosureError("formal process environment differs from the exact frozen map")
    expected_sys_path = [
        str(SHADOW_SOURCE_ROOT),
        "/home/alex/miniconda3/lib/python312.zip",
        "/home/alex/miniconda3/lib/python3.12",
        "/home/alex/miniconda3/lib/python3.12/lib-dynload",
        "/home/alex/Documents/prospect/.venv/lib/python3.12/site-packages",
    ]
    if sys.path != expected_sys_path:
        raise RuntimeClosureError("sanitized formal sys.path differs")
    for name, module in tuple(sys.modules.items()):
        if name.startswith("prospect") or name.startswith("bench.multimodal_"):
            raise RuntimeClosureError(f"forbidden project/historical module was imported: {name}")
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str):
            candidate = Path(module_file).absolute()
            if (
                candidate.is_relative_to(REPO_ROOT)
                and not candidate.is_relative_to(SHADOW_SOURCE_ROOT)
                and not candidate.is_relative_to(VENV_PREFIX)
            ):
                raise RuntimeClosureError(f"live project module escaped the shadow closure: {name}")
    chain = _validate_symlink_chain()
    dependency_hashes = {
        "numpy_init": _sha256_file(NUMPY_INIT),
        "numpy_multiarray": _sha256_file(MULTIARRAY),
        "numpy_umath_linalg": _sha256_file(UMATH_LINALG),
        "openblas": _sha256_file(OPENBLAS),
        "pyvenv_cfg": _sha256_file(VENV_PREFIX / "pyvenv.cfg"),
        "python_terminal": _sha256_file(BASE_EXECUTABLE),
    }
    expected_hashes = {
        "numpy_init": NUMPY_INIT_SHA256,
        "numpy_multiarray": MULTIARRAY_SHA256,
        "numpy_umath_linalg": UMATH_LINALG_SHA256,
        "openblas": OPENBLAS_SHA256,
        "pyvenv_cfg": PYVENV_CFG_SHA256,
        "python_terminal": PYTHON_TERMINAL_SHA256,
    }
    if dependency_hashes != expected_hashes:
        raise RuntimeClosureError("runtime dependency hashes differ")
    if Path(np.__file__).absolute() != NUMPY_INIT:
        raise RuntimeClosureError("loaded NumPy package path differs")
    import numpy.linalg._umath_linalg as loaded_linalg

    loaded_multiarray = sys.modules.get("numpy._core._multiarray_umath")
    if (
        loaded_multiarray is None
        or Path(cast(str, getattr(loaded_multiarray, "__file__", ""))).absolute() != MULTIARRAY
    ):
        raise RuntimeClosureError("loaded NumPy multiarray object differs")
    if Path(loaded_linalg.__file__).absolute() != UMATH_LINALG:
        raise RuntimeClosureError("loaded NumPy linalg object differs")
    if os.readlink("/proc/self/exe") != str(BASE_EXECUTABLE):
        raise RuntimeClosureError("kernel executable identity differs")
    manifests = dependency_manifests()
    closures = {name: _closure_aggregate(manifest) for name, manifest in manifests.items()}
    canary = svd_canary()
    expected_canary = {
        "bundle_sha256": CANARY_BUNDLE_SHA256,
        "input_sha256": CANARY_INPUT_SHA256,
        "minimum_singular_gap": CANARY_MIN_GAP,
        "relative_reconstruction_error": CANARY_RELATIVE_RECONSTRUCTION,
        "s_sha256": CANARY_S_SHA256,
        "schema_version": "lcv001-svd-canary-v1",
        "shape": [512, 256],
        "u_sha256": CANARY_U_SHA256,
        "vh_sha256": CANARY_VH_SHA256,
    }
    if canary != expected_canary:
        raise RuntimeClosureError("SVD canary differs")
    loaded = _loaded_openblas_paths()
    if loaded != (str(OPENBLAS),):
        raise RuntimeClosureError(f"loaded OpenBLAS object differs: {loaded}")
    openblas = _openblas_observation()
    if openblas != {"config": OPENBLAS_CONFIG, "core": OPENBLAS_CORE, "parallel": 1, "threads": 1}:
        raise RuntimeClosureError(f"OpenBLAS runtime observation differs: {openblas}")
    return {
        "base_executable": base_executable,
        "base_prefix": sys.base_prefix,
        "canary": canary,
        "dependency_hashes": dependency_hashes,
        "dependency_closures": closures,
        "dependency_manifests": manifests,
        "executable": sys.executable,
        "loaded_openblas": list(loaded),
        "numpy": np.__version__,
        "numpy_file": str(Path(np.__file__).absolute()),
        "openblas": openblas,
        "platform": platform.platform(),
        "prefix": sys.prefix,
        "exec_prefix": sys.exec_prefix,
        "base_exec_prefix": sys.base_exec_prefix,
        "python": platform.python_version(),
        "schema_version": SCHEMA_VERSION,
        "status": "runtime_closure_verified",
        "symlink_chain": chain,
        "environment": expected_environment,
        "thread_environment": dict(sorted(THREAD_ENVIRONMENT.items())),
    }


def frozen_environment() -> dict[str, str]:
    """Return the small environment used for formal and semantic child processes."""

    output = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
    }
    output.update(THREAD_ENVIRONMENT)
    return output


__all__ = [
    "LEXICAL_PYTHON",
    "RuntimeClosureError",
    "frozen_environment",
    "observe_and_validate",
    "dependency_closures",
    "dependency_manifests",
    "svd_canary",
    "validate_dependency_manifests",
]
