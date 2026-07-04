"""P9-002 ablation harness: the leave-one-out marginal control value of each component
of the composed agent (ADR-0008).

Disabling a load-bearing part must measurably hurt. A part that can be removed with no
loss — or that *helps* when removed — is a **finding**, not a silent pass: either the
component is dead weight (a design shortcoming) or the E2E gate cannot see its value (a
test shortcoming). These are pure functions over the control returns the P9 gate already
measures; `check_p9` gates on the clearly load-bearing component and records the table.

For the P9 control loop the ablatable components are exactly `planning`, `retrieval`,
and `exploit_penalty` (the curriculum's exploit-mode coefficient) — hierarchy and skills
are not wired into this loop, so there is nothing to ablate there.
"""
from __future__ import annotations

MARGIN = 5.0  # a marginal within +/-MARGIN return points counts as negligible (a finding)


def marginals(full: float, ablated: dict[str, float]) -> dict[str, float]:
    """Leave-one-out marginal value of each component: `full - (return with it off)`.
    Positive ⇒ the component earns its place; negative ⇒ it HURTS control (a finding);
    ~0 ⇒ dead weight (a finding)."""
    return {name: full - r for name, r in ablated.items()}


def classify(marginal: float, margin: float = MARGIN) -> str:
    """Label a marginal: load-bearing (helps), harmful (hurts), or negligible."""
    if marginal > margin:
        return "load-bearing"
    if marginal < -margin:
        return "harmful"
    return "negligible"
