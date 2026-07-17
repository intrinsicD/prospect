"""Isolated SVD-canary child used by LCV-001 runtime negative controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final, TextIO

RUNTIME_CLASSIFICATION: Final = "invalid_LCV001_runtime"
RUNTIME_TRANSPORT_EXIT: Final = 70


def _json_line(value: object, *, compact: bool = True) -> str:
    separators = (",", ":") if compact else None
    return json.dumps(value, sort_keys=True, separators=separators, allow_nan=False) + "\n"


def _write_line(payload: str, stream: TextIO) -> None:
    if stream.write(payload) != len(payload):
        raise OSError("canary CLI stream made a short write")
    stream.flush()


def _safe_exception_text(error: BaseException) -> str:
    try:
        rendered = str(error)
    except BaseException:
        rendered = "<exception text unavailable>"
    return rendered[-1000:]


def _write_runtime_failure(error: BaseException) -> int:
    payload = _json_line(
        {
            "classification": RUNTIME_CLASSIFICATION,
            "error": (
                f"runtime sensitivity canary transport failed: {type(error).__name__}: "
                f"{_safe_exception_text(error)}"
            ),
        },
        compact=False,
    )
    try:
        _write_line(payload, sys.stderr)
    except (OSError, ValueError, KeyboardInterrupt):
        return RUNTIME_TRANSPORT_EXIT
    return RUNTIME_TRANSPORT_EXIT


def main() -> int:
    try:
        if len(sys.argv) != 2:
            raise ValueError("canary probe requires one explicit site-packages path")
        shadow_source = Path(__file__).absolute().parents[2]
        sys.path[:] = [
            str(shadow_source),
            "/home/alex/miniconda3/lib/python312.zip",
            "/home/alex/miniconda3/lib/python3.12",
            "/home/alex/miniconda3/lib/python3.12/lib-dynload",
            sys.argv[1],
        ]
        from bench.sealed_lineage_verifier import runtime_probe

        payload = _json_line(runtime_probe.svd_canary())
        try:
            _write_line(payload, sys.stdout)
        except (OSError, ValueError) as output_error:
            return _write_runtime_failure(output_error)
    except OSError as host_error:
        return _write_runtime_failure(host_error)
    except KeyboardInterrupt as interrupt:
        return _write_runtime_failure(interrupt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
