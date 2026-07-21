"""Explicit assurance boundary for WM-001 protocol 1.14 evidence."""

from __future__ import annotations

from collections.abc import Mapping

TRUST_MODEL_ID = "prospect.wm001.trust-model.v1"
TRUST_MODEL_STATEMENT = (
    "The kernel, filesystem implementation, base interpreter and standard "
    "library, invoking account, and every process able to write the repository, "
    "isolated environment, or results roots are trusted. A WM-001 run has "
    "exclusive use of those paths; no cooperating or malicious same-principal "
    "writer may run concurrently. Every conforming sealed-runtime invocation "
    "acquires one repository-global nonblocking advisory lock for its complete "
    "child lifetime; QA-only preformal commands remain covered by the "
    "exclusive-use assumption. The lock coordinates trusted harness processes "
    "but is not a security boundary against noncooperating same-principal "
    "writers. Hashes, descriptor checks, pre/post inventories, and no-replace "
    "publication detect accidental or persistent drift and application-level "
    "overwrite; they do not provide tamper resistance against the account or "
    "environment owner, transient mutate-and-restore attacks, privileged "
    "actors, or a compromised kernel. Immutable means protocol-level "
    "append-only/no-replace evidence under this trust boundary, not fs-verity, "
    "read-only media, a TPM, an external signer, or external attestation."
)

ASSURANCE: dict[str, object] = {
    "trust_model_id": TRUST_MODEL_ID,
    "tamper_resistant": False,
    "external_attestation": False,
    "exclusive_path_use_required": True,
}


def assurance_record() -> dict[str, object]:
    """Return a fresh canonical assurance record."""

    return dict(ASSURANCE)


def require_assurance(value: object, *, label: str) -> dict[str, object]:
    """Reject evidence that omits or overstates the fixed assurance boundary."""

    if not isinstance(value, Mapping) or dict(value) != ASSURANCE:
        raise ValueError(f"{label} differs from {TRUST_MODEL_ID}")
    return dict(value)


__all__ = [
    "ASSURANCE",
    "TRUST_MODEL_ID",
    "TRUST_MODEL_STATEMENT",
    "assurance_record",
    "require_assurance",
]
