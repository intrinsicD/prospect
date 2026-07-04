"""Eval bodies for gates & sentinels (P0-006).

Modules here self-register their checks via `@gate_check(phase)` /
`@sentinel_check(name)` on import; criteria stay as data in `bench.gates`.
Add an import below when a new eval module lands.
"""
from __future__ import annotations

from . import (  # noqa: F401
    p0_scaffold,
    p1_world_model,
    p2_planner,
    p3_replay,
    p3_voe,
    p4_skills,
    p5_options,
    p6_codec,
    p7_continual,
)
