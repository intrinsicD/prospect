# WM-001 v1.5.0 development failure review

## Disposition

WM-001 protocol 1.5.0 is retired without a qualifying development rehearsal
and without a formal attempt. Development attempt 4 is a preserved,
unfinalized failed producer. It cannot support a claim that Prospect learned,
improved, retained an improvement, or survived restart.

No K3–K6 performance value from the failed attempt was inspected, summarized,
thresholded, or used to modify the scientific system.

## What happened

The prospective runtime seal and result-free CUDA inventory rehearsal both
passed. Development then began with the first declared development seed. It
completed collection, training, the declared A/B control evaluations, and
checkpoint creation, but stopped when restart parity performed its first
post-environment live-closure recomputation.

The producer recorded:

- status: `failed`;
- error: `RuntimeError: live runtime closure differs from its pre-import bootstrap seal`;
- producer manifest SHA-256:
  `f79a798f87d8f057060f0d3edc6b1eb676cab449dfa026c69ddaef4a93705fe6`;
- producer manifest link count: one;
- deterministic outer completion: absent;
- `result.json`: absent;
- development closure: absent; and
- `formal-launch-v1.5.0.json`: absent.

The attempt-4 runtime seal remains finalized with SHA-256
`0c958fc0f890d614ea11dd34898038e9dfd05abb337c92d8bd6bd8cb8b4a181b`.
Its sealed result-free rehearsal passed three repeated inventories with
conformance SHA-256
`f429cab3ba211c926940ff3ede49c8126046bfac0c466d5c7d2284251aabfc9b`.

## Root cause

Gymnasium 0.29.1 is imported lazily when the first real Pendulum environment
is constructed. On Linux its package initializer adds exactly:

```text
PYGAME_HIDE_SUPPORT_PROMPT=hide
SDL_AUDIODRIVER=dsp
```

An isolated `-I -S -B` reproduction starting from the five-variable sealed
environment confirmed that these were the only persistent additions.
Importing the experiment configuration did not cross this lazy boundary, and
the result-free inventory rehearsal exercised closure construction but not a
Gymnasium import. The omission therefore escaped both prospective checks.

The failure is in runtime-environment custody, not in the world model,
learning algorithm, optimizer, controller, budgets, controls, metrics,
thresholds, or scientific gates.

## Why v1.5 is not retried

The general protocol preserves numbered development attempts before closure,
but the complete v1.5 execution contract prospectively designated attempt 4
as its only outcome-producing qualification and explicitly disallowed a
retry. Attempt 4 crossed that boundary. Authorizing an attempt-5 sibling only
after observing the failure would be a retroactive relaxation of the operator
contract.

The repair therefore moves to a new protocol version with:

- the four scientific source files and scientific-block digest unchanged;
- both Gymnasium variables fixed from process start;
- a result-free rehearsal that crosses the lazy Gymnasium construction
  boundary and then recomputes the exact live closure;
- fresh derivation-domain seeds;
- fresh canonical result and marker namespaces; and
- an explicit rule defining when a development attempt is consumed.
