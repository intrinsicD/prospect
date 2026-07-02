# ADR-0007 — Hard quality floor: vault + frozen probes + rollback-on-regression

**Status:** Accepted

## Context
ADR-0005 gates *phase advancement*. ADR-0006 sentinels *detect collapse* — but a
sentinel only reports health; it has **no actuator**. The moment the system trains
continually (P1 onward) and especially once it changes *itself* (P3 replay, P7
consolidation, ADR-0009 growth), the loop amplifies whatever failure it is given:

> a self-improving system amplifies whatever loop it is given; if the loop has silent
> failure modes, self-improvement optimizes the failure.

So detection is not enough — we need a mechanism that both **measures** capability on
a frozen held-out probe and **acts** to prevent silent regression. External evidence:
a sibling multimodal-harness project (OmniLatent) built exactly this — a
content-addressed checkpoint vault + frozen probe sets + rollback-on-regression — and
found it to be its single most safety-relevant property. Prospect already wrote down
*what* to watch (ADR-0006); this ADR adds *what to do when a watch trips*.

## Decision
Introduce a **hard quality floor** as harness infrastructure (`bench/`, never core):

1. **Frozen probe sets.** Each component owns a small, never-trained-on, hashed
   held-out probe. Score with **≥2 uncorrelated metrics**, at least one being
   **calibrated surprise (NLL)** — never a single easily-gamed number
   (reward-hacking guard). Prospect metrics, not pixels: NLL, epistemic/aleatoric
   separation, retention, plasticity.
2. **Content-addressed checkpoint vault.** Best-so-far weights + auxiliary state
   (EMA teacher, Fisher/SI, replay) stored SHA-256-addressed; duplicate snapshots are
   free; non-`best` snapshots LRU-evicted under a disk cap.
3. **Rollback-on-regression.** Every `EVAL_EVERY` steps, score the student. If it is
   worse than the vault best beyond a **per-metric tolerance** on *any* probe metric,
   restore the best checkpoint, reduce the learning rate, and log the regression with
   full provenance (which data, which step). Better-than-best atomically promotes.
4. **Sentinel-paired eligibility.** A checkpoint is floor-eligible only if its
   ADR-0006 sentinels are healthy — a collapsed-but-low-loss model can never be
   promoted to `best`. The floor and the sentinels are one gate, not two.
5. **Drift budget.** Pause and emit a report if cumulative EMA-vs-init weight L2
   crosses a threshold — catches a runaway self-training loop before it corrupts
   weights.

A **floor is a distinct construct from a sentinel**: a sentinel detects collapse and
reports health; the floor measures regression against a frozen probe and *acts*
(restore best-known checkpoint). It is enforced in `bench/gates.py` as a third leg of
the composite gate — a phase passes only if **capability** passes AND every applicable
**sentinel** is healthy AND the **quality floor** is satisfied. Active **from P1**
(put the floor down *before* the system starts changing itself).

## Consequences
- (+) Capability can rise with a hard guarantee it will not silently fall — the
  enabling safety property for continual improvement (R7) and self-directed growth
  (ADR-0009).
- (+) Gives the ADR-0006 sentinels the actuator they lack: detection now triggers
  rollback, not just a red light.
- (+) The ≥2-uncorrelated-metric rule makes reward-hacking a *gate failure*, not a
  surprise found later.
- (−) Frozen probes + vault add evaluation and storage cost to every gate; per-metric
  tolerances are hyperparameters to calibrate.
- (−) Too-tight tolerance thrashes (constant rollbacks, no learning); too-loose lets
  slow drift through. Tolerance is per-metric and explicit, never global.
