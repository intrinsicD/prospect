# E-series runtime substrate audit

**Audited:** 2026-07-17<br>
**Scope:** the first linked-lifecycle reference implementation<br>
**Resume contract:** episode boundary only

Prospect owns the epistemic domain records, causal linkage, decision-value
decomposition, lifecycle orchestration, and independent evaluator. It should not
own another tensor container, replay algorithm, optimizer framework, or neural
training stack.

## Selected released substrate

| Package | Pinned version | License | Used for |
|---|---:|---|---|
| PyTorch | 2.9.x | BSD-3-Clause | tensor and future model/optimizer state |
| TorchRL | 0.13.3 | MIT | `TensorDictReplayBuffer` and `LazyTensorStorage` |
| TensorDict | 0.13.0 | BSD | typed batched tensor payloads and serialization substrate |

The packages are isolated in the optional `runtime` dependency group. The finite
Bayesian semantic oracle and E-series contract tests do not require them.

TorchRL describes itself as a beta feature for which breaking changes can still
occur, so Prospect pins the exact TorchRL and TensorDict releases rather than
floating on a minor range. The adapter imports them lazily and keeps TensorDict
types out of `prospect.domain`.

Primary references:

- [TorchRL 0.13.3 release and MIT license](https://pypi.org/project/torchrl/)
- [TorchRL replay-buffer API](https://docs.pytorch.org/rl/stable/reference/data_replaybuffers.html)
- [TensorDict 0.13 overview](https://docs.pytorch.org/tensordict/stable/overview.html)
- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)

## Adapter boundary

`TensorDictExperienceReplay` is a lossy capacity-bounded sampling index. A caller
supplies the domain-to-TensorDict codec; the replay buffer never becomes the
canonical evidence ledger. `InMemoryExperienceStore` and `EpistemicLedger` retain
the immutable linked records required for scientific custody.

The runtime does not yet adopt a TorchRL collector. The reference environment is a
small exact finite problem, and introducing an asynchronous tensor collector there
would add machinery without exercising its intended batched use. A neural/control
backend should add the collector adapter when it can preserve the exact
`DecisionRecord -> ExperienceEvent` identity and prove that it does not duplicate
the authoritative runtime collector.

## Checkpoint discrepancy and temporary coordinator

The installed, pinned TorchRL 0.13.3 release exposes replay
`dumps`/`loads` and `state_dict`/`load_state_dict`, but:

```text
import torchrl.checkpoint
-> ModuleNotFoundError
```

TorchRL's current `main` documentation already describes a manifest-driven
`torchrl.checkpoint.Checkpoint`, including replay and global RNG adapters, but that
module is not present in the released 0.13.3 package. See the
[unreleased checkpoint documentation](https://docs.pytorch.org/rl/main/reference/checkpoint.html).

Until that coordinator is released and compatibility-tested, Prospect uses a
small `CheckpointCoordinator` that:

- treats each component's state as opaque bytes;
- writes one canonical manifest with component version, size, media type, and
  SHA-256;
- rejects missing, extra, oversized, malformed, or corrupted members before
  invoking any restorer; and
- atomically replaces a local episode-boundary ZIP bundle.

This is integration glue, not a proposed learning contribution. Its deletion
trigger is a released TorchRL checkpoint coordinator that passes Prospect's strict
component-coverage, corruption, RNG, replay-order, and restart-equivalence tests.
Prospect may then retain only a domain-manifest adapter for evidence identities the
upstream format does not know about.

## State that a full checkpoint must cover

The coordinator is format-agnostic, but an E-series retention claim is invalid
unless the manifest includes every stateful category used by that agent:

- model, target model, optimizer, scheduler, and normalization;
- calibration and information-value estimator;
- replay storage, sampler, writer position, priorities, and sample RNG;
- current belief/filter state at the declared boundary;
- policy/regulator and knowledge state;
- configuration, source, model, representation, policy, and calibration versions;
- run/episode/update/identity counters; and
- Python, NumPy, PyTorch CPU, and every relevant device RNG state.

The first contract is episode-boundary restoration. Exact mid-episode resume
requires environment state, pending intentions, recurrent belief state, and
side-effect reconciliation and is not claimed.
