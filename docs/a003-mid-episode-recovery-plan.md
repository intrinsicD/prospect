# A-003 mid-episode recovery plan

Status: queued; planning only.

## Dependency

Requires accepted A-001 durable recovery and A-002 lifecycle ordering. Start
with a deterministic, fully snapshot-capable simulator before considering
external environments.

## Falsifiable objective

After interruption at any declared point in one simulator interaction, Prospect
restores the environment, recurrent belief, pending intention/execution,
experience custody, side-effect disposition, and RNG state and then produces
the same remaining trace as an uninterrupted reference run.

## Gates

1. Define the exact resume boundary and component inventory for
   pre-decision, post-decision, post-execution, post-observation, and
   post-storage states.
2. Bind environment snapshot semantics, action idempotency key, side-effect
   acknowledgment, all RNG streams, recurrent state, clock, and pending record
   identities.
3. Inject process death at every declared boundary and require no duplicated or
   omitted environment action or canonical experience.
4. For a deterministic simulator, require exact remaining observations,
   predictions, actions, rewards, termination state, and identities versus the
   uninterrupted run.
5. Reject an environment or tool that cannot prove snapshot or idempotent
   execution rather than claiming recovery through inference.
6. Require repeated restore attempts to converge to one identical terminal
   state.

## Exclusions

No claim for arbitrary real-world actuators, payments, messages, remote APIs,
or other non-idempotent effects. No cross-machine or adversarial-kernel claim.

## Intended repository surface

Resume-boundary protocols under `src/prospect/domain/`, coordinator work under
`src/prospect/runtime/`, `tests/test_a003_mid_episode_recovery.py`, and one
snapshot-capable simulator fixture under `bench/assurance/`.

## Next action

Write the five-boundary state inventory and map which current components can
already dump/restore exact bytes. Mark every missing environment, recurrent,
pending-action, side-effect, clock, and RNG component explicitly.
