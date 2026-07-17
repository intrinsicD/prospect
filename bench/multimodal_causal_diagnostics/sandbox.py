"""Linux Landlock/seccomp boundary for the MM-009 source-only predictor.

The formal shadow runtime imports this file directly, before importing NumPy or any
scientific module.  It intentionally depends only on the Python standard library.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Final

LANDLOCK_ABI_REQUIRED: Final = 6

_SYS_LANDLOCK_CREATE_RULESET: Final = 444
_SYS_LANDLOCK_ADD_RULE: Final = 445
_SYS_LANDLOCK_RESTRICT_SELF: Final = 446
_LANDLOCK_CREATE_RULESET_VERSION: Final = 1
_LANDLOCK_RULE_PATH_BENEATH: Final = 1

_FS_EXECUTE: Final = 1 << 0
_FS_WRITE_FILE: Final = 1 << 1
_FS_READ_FILE: Final = 1 << 2
_FS_READ_DIR: Final = 1 << 3
_FS_MAKE_REG: Final = 1 << 8
_FS_TRUNCATE: Final = 1 << 14
_FS_HANDLED: Final = (1 << 15) - 1
_FS_READ: Final = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
_FS_OUTPUT: Final = _FS_READ | _FS_WRITE_FILE | _FS_MAKE_REG | _FS_TRUNCATE

_NET_BIND_TCP: Final = 1 << 0
_NET_CONNECT_TCP: Final = 1 << 1
_NET_HANDLED: Final = _NET_BIND_TCP | _NET_CONNECT_TCP

_PR_SET_NO_NEW_PRIVS: Final = 38
_SECCOMP_ACT_ALLOW: Final = 0x7FFF0000
_SECCOMP_ACT_ERRNO: Final = 0x00050000

_BLOCKED_SYSCALLS: Final = (
    # A custody child must not replace itself with, or delegate to, any executable
    # outside the sealed Python import closure.  It must also remain in the fresh
    # process group owned by the supervisor so cancellation is complete.
    "execve",
    "execveat",
    "setsid",
    "setpgid",
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "sendto",
    "recvfrom",
    "sendmsg",
    "recvmsg",
    "sendmmsg",
    "recvmmsg",
    "shutdown",
    "getsockopt",
    "setsockopt",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "kcmp",
    "pidfd_getfd",
    "open_by_handle_at",
    "name_to_handle_at",
    "mount",
    "umount2",
    "pivot_root",
    "chroot",
    "setns",
    "unshare",
    "bpf",
    "perf_event_open",
    "userfaultfd",
    "keyctl",
    "add_key",
    "request_key",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
)


class SandboxUnavailable(RuntimeError):
    """Raised when the host cannot establish the frozen source-only boundary."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = (("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64))


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = (("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32))


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def landlock_abi() -> int:
    """Return the highest supported ABI, failing closed on this frozen platform."""

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SandboxUnavailable("MM-009 Landlock syscall pins require Linux x86_64")
    libc = _libc()
    value = int(
        libc.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    )
    if value < 0:
        error = ctypes.get_errno()
        raise SandboxUnavailable(f"Landlock ABI query failed: errno={error}")
    return value


def _checked_existing(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise SandboxUnavailable(f"{label} must be absolute")
    if candidate.is_symlink() or not candidate.exists():
        raise SandboxUnavailable(f"{label} must exist and not be a symlink: {candidate}")
    return candidate


def _add_path_rule(libc: ctypes.CDLL, ruleset_fd: int, path: Path, access: int) -> None:
    descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = _PathBeneathAttr(access, descriptor)
        status = int(
            libc.syscall(
                _SYS_LANDLOCK_ADD_RULE,
                ruleset_fd,
                _LANDLOCK_RULE_PATH_BENEATH,
                ctypes.byref(rule),
                0,
            )
        )
        if status != 0:
            error = ctypes.get_errno()
            raise SandboxUnavailable(f"Landlock path rule failed for {path}: errno={error}")
    finally:
        os.close(descriptor)


def install_landlock(
    readable_paths: Sequence[str | Path],
    writable_directory: str | Path,
) -> int:
    """Install the frozen deny-by-default filesystem/TCP policy and return its ABI."""

    abi = landlock_abi()
    if abi < LANDLOCK_ABI_REQUIRED:
        raise SandboxUnavailable(f"Landlock ABI {abi} is below required ABI {LANDLOCK_ABI_REQUIRED}")
    readable = tuple(_checked_existing(path, "readable path") for path in readable_paths)
    output = _checked_existing(writable_directory, "writable directory")
    if not output.is_dir():
        raise SandboxUnavailable("writable path must be a directory")
    if len(set(readable)) != len(readable):
        raise SandboxUnavailable("readable Landlock paths must be unique")
    if output in readable:
        raise SandboxUnavailable("writable directory cannot also be a read-only rule")

    libc = _libc()
    attributes = _RulesetAttr(_FS_HANDLED, _NET_HANDLED)
    ruleset_fd = int(
        libc.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(attributes),
            ctypes.sizeof(attributes),
            0,
        )
    )
    if ruleset_fd < 0:
        error = ctypes.get_errno()
        raise SandboxUnavailable(f"Landlock ruleset creation failed: errno={error}")
    try:
        for path in readable:
            _add_path_rule(
                libc,
                ruleset_fd,
                path,
                _FS_READ if path.is_dir() else _FS_READ_FILE,
            )
        _add_path_rule(libc, ruleset_fd, output, _FS_OUTPUT)
        if int(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) != 0:
            error = ctypes.get_errno()
            raise SandboxUnavailable(f"PR_SET_NO_NEW_PRIVS failed: errno={error}")
        if int(libc.syscall(_SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)) != 0:
            error = ctypes.get_errno()
            raise SandboxUnavailable(f"Landlock restriction failed: errno={error}")
    finally:
        os.close(ruleset_fd)
    return abi


def install_seccomp() -> None:
    """Deny network, cross-process, namespace, and kernel-expansion syscalls."""

    try:
        library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as error:
        raise SandboxUnavailable("libseccomp.so.2 is unavailable") from error
    library.seccomp_init.argtypes = (ctypes.c_uint32,)
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    )
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = (ctypes.c_void_p,)
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = (ctypes.c_void_p,)
    library.seccomp_release.restype = None

    context = library.seccomp_init(_SECCOMP_ACT_ALLOW)
    if not context:
        raise SandboxUnavailable("seccomp_init failed")
    try:
        action = _SECCOMP_ACT_ERRNO | errno.EACCES
        for name in _BLOCKED_SYSCALLS:
            number = int(library.seccomp_syscall_resolve_name(name.encode("ascii")))
            if number < 0:
                raise SandboxUnavailable(f"seccomp cannot resolve syscall {name}")
            status = int(library.seccomp_rule_add(context, action, number, 0))
            if status != 0:
                raise SandboxUnavailable(f"seccomp rule failed for {name}: errno={-status}")
        status = int(library.seccomp_load(context))
        if status != 0:
            raise SandboxUnavailable(f"seccomp_load failed: errno={-status}")
    finally:
        library.seccomp_release(context)


def close_inherited_descriptors() -> None:
    """Close every non-stdio descriptor before constructing the policy."""

    upper = 65_536
    try:
        soft_limit = int(os.sysconf("SC_OPEN_MAX"))
    except (OSError, ValueError):
        soft_limit = upper
    os.closerange(3, min(max(soft_limit, 3), upper))


def enter_source_only_sandbox(
    readable_paths: Sequence[str | Path],
    writable_directory: str | Path,
) -> int:
    """Close authority, install the syscall filter, then restrict filesystem access."""

    close_inherited_descriptors()
    install_seccomp()
    return install_landlock(readable_paths, writable_directory)


__all__ = [
    "LANDLOCK_ABI_REQUIRED",
    "SandboxUnavailable",
    "close_inherited_descriptors",
    "enter_source_only_sandbox",
    "install_landlock",
    "install_seccomp",
    "landlock_abi",
]
