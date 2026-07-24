# EXT-001 independent external-evaluation plan

Status: queued; planning only, not an evaluation authority.

## Dependency

Begin after WM-003B or a later externally recognizable benchmark result is
formally accepted and A-004 has demonstrated the package's bounded
cross-machine restore contract. Freeze the accepted agent snapshot before
selecting or opening external test outcomes. Reuse A-004's portability manifest
and conformance checks; do not redefine them here.

## Falsifiable objective

An independently operated evaluator can run the frozen Prospect snapshot and
reproduce its bounded capability result under a declared interface, submission,
environment, data, compute, and planning budget.

## Gates

1. Bind the evaluator version, task split, scoring code, allowed dependencies,
   hardware envelope, baseline versions, submission limit, and tuning policy.
2. Export one immutable source/container identity and checkpoint digest; the
   evaluator must report the same identities.
3. Run the exact local conformance suite against the exported package before
   consuming a submission.
4. Retain the evaluator's independently issued receipt, raw per-task scores,
   failures, resource use, and timestamp.
5. Compare only against baselines using compatible observations, environment
   steps, compute, planner calls, and test access; label incompatible published
   numbers separately.
6. Require a second verifier to recompute aggregates and confirm there was no
   post-result tuning or package substitution.

## Exclusions

No algorithm, prompt, model, checkpoint, or hyperparameter changes after test
outcomes are opened. One external arena cannot establish general maturity,
novelty, or state-of-the-art superiority. Independent execution is evidence
about portability and benchmark behavior, not hostile-kernel security.

## Intended repository surface

`bench/external_evaluation/` for adapters and local conformance, with generated
submission bundles and receipts outside version control except for compact,
reviewed summaries and public identifiers.

## Next action

For the WM-003A/WM-003B benchmark candidates, enumerate hosted or independently
runnable
evaluation options, submission rules, baseline comparability, costs, and
artifact-retention guarantees. Do not submit until an accepted snapshot exists.
