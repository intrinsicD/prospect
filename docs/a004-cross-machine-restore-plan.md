# A-004 cross-machine semantic-restore plan

Status: queued; planning only.

## Dependency

Requires A-001's stable checkpoint and recovery semantics plus two
independently provisioned execution environments. It may proceed independently
of A-002 and A-003 when only episode-boundary restore is tested.

## Falsifiable objective

A checkpoint produced on one declared machine/dependency closure restores on a
different declared machine and preserves prospectively bounded predictive,
decision, and executed-behavior semantics without silent backend or precision
substitution.

## Gates

1. Bind source and destination CPU/GPU, driver, interpreter, dependency,
   precision, device-selection, and environment identities.
2. Separate invariants requiring exact equality—schema, versions, ancestry,
   component digests, action identities—from numerical and behavioral metrics
   requiring prospectively frozen tolerances.
3. Require the destination to disclose all conversions, device fallbacks, and
   unsupported operators before evaluation.
4. Recompute fixed prediction, planning, and executed evaluation probes on both
   machines and apply paired semantic-parity gates.
5. Repeat destination restoration in a fresh environment and require stable
   results within the same bounds.
6. Publish both passing and unsupported platform pairs; unsupported restore
   must fail closed, not silently weaken the contract.

## Exclusions

No promise of byte-identical floating-point kernels across hardware, equal
throughput, general deployment portability, hosted-benchmark superiority, or
resistance to a compromised machine.

## Intended repository surface

`bench/assurance/cross_machine_restore/`, compact machine manifests and
schemas, and `tests/test_a004_portability_contract.py`. Large checkpoints and
machine-specific outputs remain external.

## Next action

Classify every WM-001 checkpoint component as exact-portable,
conversion-required, hardware-bound, or unsupported, then choose one realistic
source/destination pair and define tolerances without opening its outcomes.
