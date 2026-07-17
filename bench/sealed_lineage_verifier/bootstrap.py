"""Stdlib-only bootstrap for the copied LCV-001 shadow package."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    shadow_source = Path(__file__).absolute().parents[2]
    allowed = [
        str(shadow_source),
        "/home/alex/miniconda3/lib/python312.zip",
        "/home/alex/miniconda3/lib/python3.12",
        "/home/alex/miniconda3/lib/python3.12/lib-dynload",
        "/home/alex/Documents/prospect/.venv/lib/python3.12/site-packages",
    ]
    sys.path[:] = allowed
    from bench.sealed_lineage_verifier import experiment

    if sys.path != allowed:
        raise RuntimeError("LCV-001 sanitized import path changed")
    return experiment.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
