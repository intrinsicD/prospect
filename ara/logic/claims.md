# Claims

## C01: The BridgeControl transition stack is a material failure source
- **Statement**: On the frozen BC-001 balanced cell with matched planner compute,
  replacing learned recursive ensemble-mean rollouts with exact target-refresh
  transitions improves mean return in 8/8 formal seeds, closes 79.35% of the paired
  oracle-return gap, and raises success from 6.25% to 84.375%. This identifies the
  transition-mean/recursive-refresh stack, not representation capacity alone.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The sealed semantic verifier fails, or a protocol-matched
  rerun does not satisfy the frozen 7/8 direction, 20% gap-closure, and fixed-bank
  regret criteria.
- **Proof**: [`bench/oracle_ladder_v2/results/OL-002/OL-002-results.json`,
  `bench/oracle_ladder_v2/results/OL-002/OL-002-report.md`,
  `ara/evidence/oracle-ladder-ol002-2026-07-14.md`]
- **Dependencies**: []
- **Tags**: BridgeControl, transition stack, recursive rollout, simulator oracle
- **From staging**: O11

## C02: Learned reward is a second material BridgeControl bottleneck
- **Statement**: On exact online-refresh BridgeControl paths, replacing learned reward
  with exact reward improves mean return in 8/8 formal seeds, closes 100% of the
  remaining paired oracle-return gap, and raises success from 84.375% to 100%. This
  identifies learned reward composed with online encoding, not reward-head weights
  alone.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The sealed semantic verifier fails, or a protocol-matched
  rerun does not satisfy the frozen 7/8 direction, 20% gap-closure, and fixed-bank
  regret criteria.
- **Proof**: [`bench/oracle_ladder_v2/results/OL-002/OL-002-results.json`,
  `bench/oracle_ladder_v2/results/OL-002/OL-002-report.md`,
  `ara/evidence/oracle-ladder-ol002-2026-07-14.md`]
- **Dependencies**: []
- **Tags**: BridgeControl, reward stack, online encoding, simulator oracle
- **From staging**: O12
