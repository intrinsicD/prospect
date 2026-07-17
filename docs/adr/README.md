# Architecture Decision Records

Short, dated records of the decisions the whole design leans on. One decision per
file. If a change would contradict an accepted ADR, **amend or supersede the ADR
first** — a 15-line ADR is cheaper than silent architectural drift.

Format: Status · Context · Decision · Consequences.

ADR-0014 starts the E-series architecture. Where it conflicts with ADR-0002 or
ADR-0007, ADR-0014 governs. P0–P14 results remain frozen legacy-v1 evidence and are
not retroactively treated as E-series lifecycle evidence.

| ADR | Decision | Status |
|-----|----------|--------|
| 0001 | Latent predictive world model as the spine | Accepted; amended by 0014 |
| 0002 | Prediction error (VoE) as the single unifying signal | Superseded by 0014 (legacy-v1) |
| 0003 | Hierarchical planning via a jumpy option-model | Accepted |
| 0004 | Three-tier knowledge; retrieval/tools as uncertainty-gated actions | Accepted; amended by 0014 |
| 0005 | Benchmark-gated incremental delivery | Accepted |
| 0006 | Representation & uncertainty integrity (collapse prevention) | Accepted; amended by 0014 |
| 0007 | Arbitration of the epistemic signal (explore/exploit modes) | Superseded by 0014 (legacy-v1) |
| 0008 | Whole-system validation (integration, ablation, generalization) | Accepted; amended by 0014 |
| 0009 | Omni-modal seams: any modality in/out, specialized per deployment | Accepted |
| 0010 | Learning from action-free observation via latent-action inference | Accepted |
| 0011 | Optional, non-gated harder-benchmark tier (real MuJoCo control) | Accepted |
| 0012 | Imitation from observation: recover actions to reproduce a demo | Accepted |
| 0013 | Two-tier storage for research artifacts | Accepted |
| 0014 | Linked epistemic transitions and independent lifecycle claims | Accepted; implementation/evidence pending |
