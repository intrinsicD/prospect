# OL-002 BridgeControl simulator-oracle localization

## Scope and provenance

- **Date:** 2026-07-14
- **Provenance:** AI-executed after the user directed continuation of the proposed
  next experiment.
- **Scope:** Non-gated BridgeControl precursor to portfolio program P3. No production
  source, gate, task, ADR, or shipped benchmark state changed.
- **Parent evidence:** `bench/bridge_control/results/BC-001/`
- **Frozen protocol:**
  `docs/research/2026-07-14-oracle-prefix-ladder-protocol.md`
- **Administrative rerun delta:**
  `docs/research/2026-07-14-oracle-prefix-ladder-ol002-protocol.md`
- **Machine result:**
  `bench/oracle_ladder_v2/results/OL-002/OL-002-results.json`
- **Human report:**
  `bench/oracle_ladder_v2/results/OL-002/OL-002-report.md`

## Evidence identity

OL-001 completed its numeric run and deterministic semantic rerun but failed its
terminal CSV rendering check because `Path.read_text()` normalized correct CRLF bytes
to LF before comparison. The failed namespace and hashes remain preserved in
`docs/research/2026-07-14-ol001-verifier-failure.md`.

OL-002 is a fresh full rerun with the same scientific fields and an LF canonical CSV
renderer. It was frozen after OL-001 outcomes were visible, so OL-001 and OL-002 are
**one experiment, not independent evidence**, and must never be double-counted.

## Integrity gates

- Eight formal seeds (`0..7`), four fixed starts per seed.
- Every regenerated model exactly replayed its saved BC-001 baseline return, final
  state, and success result.
- Recursive-mean `k=0` wrapper parity and full exact-wrapper parity were exact for all
  eight seeds; recorded maximum action, state, return, and fixed-bank score
  differences were zero.
- Raw exact dynamics/reward solved 32/32 starts.
- The OL-002 `run` command completed its built-in full deterministic semantic verifier
  and returned `completed_localization`.

## Endpoint results

| Rung | Mean return | Success |
|---|---:|---:|
| Learned TS-infinity, penalty 0.03 | -2.770126 | 6.25% |
| Learned TS-infinity, penalty 0 | -2.660520 | 6.25% |
| Learned recursive mean, penalty 0 | -4.702069 | 6.25% |
| Exact transition, target refresh, learned reward | 3.213480 | 84.375% |
| Exact transition, online refresh, learned reward | 3.438678 | 84.375% |
| Exact transition, online refresh, exact reward | 5.273024 | 100% |
| Raw exact ceiling | 5.273024 | 100% |

## Frozen component contrasts

| Contrast | Positive seeds | Mean-return delta | Oracle gap closed | Success delta | Decision |
|---|---:|---:|---:|---:|---|
| Remove epistemic penalty | 6/8 | +0.109606 | 1.36% | 0 pp | Not material |
| Recursive mean versus TS-infinity | 2/8 | -2.041550 | -25.73% | 0 pp | Not material |
| Exact-target transition stack | 8/8 | +7.915550 | 79.35% | +78.125 pp | Material, dominant |
| Online versus target refresh | 6/8 | +0.225198 | 10.93% | 0 pp | Not material |
| Exact reward on exact-online paths | 8/8 | +1.834346 | 100% of remaining gap | +15.625 pp | Material, dominant |

The classification is `mixed:transition_stack,reward_stack`. The transition contrast
identifies the learned transition-mean/recursive-refresh stack, not representation
capacity alone. The reward contrast identifies learned reward composed with online
encoding, not reward-head weights alone.

## Prefix ladder

| Exact prefix depth | Mean return | Success | Frozen recovery rule |
|---:|---:|---:|---|
| 0 | -4.702069 | 6.25% | No |
| 1 | -0.428830 | 65.625% | No |
| 2 | 0.088528 | 53.125% | No |
| 4 | 1.336343 | 56.25% | No |
| 8 | 3.333485 | 96.875% | Yes |
| 12 | 3.213480 | 84.375% | Yes |

There is no accepted minimum recovery depth: seven of eight seed-level returns reverse
from `k=8` to `k=12`, triggering the frozen no-knee rule. The nonmonotonic curve is
evidence against a simple stable compounding-error threshold; its cause remains open.

## Artifact hashes

- `OL-002-results.json`:
  `9eecb1c382e63aec13c560b6e32223a8c0e5d73610c64bfd922b60a813e1f1df`
- `OL-002-report.md`:
  `b8590ba2186c3f58c049d2934d2f537212af4a6f02f4fe6f33808a2f9e820b16`
- `OL-002-runs.csv`:
  `44b571d6532825b57769a475b46a635a625ec32fa5b62bef38aba5dff2449e28`
- `artifact-manifest.json`:
  `e52746f6137d66b4524f3db502349005401731899e07552e513b06762227f948`
- Frozen BC-001 dataset copy:
  `9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948`

## Boundary and next question

Not run: privileged candidate injection, enlarged learned-landscape search, MuJoCo P3
replication, or production/task activation. The smallest unresolved question is why an
eight-step exact prefix outperforms the all-exact-transition/learned-reward endpoint;
a follow-up should jointly measure reward-ranking alignment and learned-tail effects at
each prefix depth before treating U-006 or a search intervention as selected.
