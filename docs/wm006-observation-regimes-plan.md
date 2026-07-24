# WM-006 learned-observation-regime plan

Status: optional and queued; planning only, not a sealed protocol.

## Dependency

Begin after WM-005 reaches a terminal disposition. Reuse WM-003A/WM-003B's
frozen vector-state benchmark as the representation ceiling and WM-005's
checkpointable belief-state design, so observation representation is the only
primary causal delta.

## Falsifiable objective

Replacing privileged state vectors with a learned pixel observation encoder
still permits the same linked agent to learn a predictive world model, improve
executed control, and retain the improvement under matched interaction and
planning budgets.

## Required controls

- privileged state-vector agent as a ceiling;
- frozen pretrained encoder;
- random frozen features with matched dimension;
- temporally shuffled frame/transition pairing; and
- pixel agent with learning disabled.

## Gates

1. Bind frame provenance, preprocessing, augmentations, encoder version,
   temporal alignment, train/validation/evaluation video or reset splits, and
   compute budget.
2. Reject duplicate-frame, adjacent-window, normalization, augmentation, and
   simulator-state leakage across splits.
3. Require the learned representation to improve held-out next-observation or
   latent-dynamics proper score over random and shuffled controls.
4. Require executed control improvement over the frozen pixel agent; offline
   feature quality alone is insufficient.
5. Report sample efficiency and compute separately from final return, with the
   state-vector ceiling clearly identified.
6. Demonstrate that the responsible encoder, world-model, optimizer, replay,
   and preprocessing state survives fresh-process restore.

The retained Perception Test data may qualify ingestion, provenance, and
temporal tests, but cannot establish closed-loop behavioral improvement.

## Exclusions

The first task adds pixels only. It does not simultaneously add audio,
language, cross-modal fusion, active acquisition, or a claim of general
multimodal intelligence.

## Intended repository surface

`bench/world_model_observation_regimes/`, `tests/test_wm006_*.py`, and reusable
observation protocols in `src/prospect/domain/` only after their
backend-neutral semantics are demonstrated.

## Next action

Audit the retained Perception Test fixtures for reusable adapter qualification,
then select one closed-loop pixel-control environment whose underlying state
version can serve as a matched ceiling.
