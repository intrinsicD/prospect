# WM-001 v1.7.0 development-closure failure review

## Disposition

WM-001 protocol 1.7.0 is retired without a qualifying development closure and
without a formal attempt. The canonical two-seed development producer
completed, its required independent audit was accepted, and both attempts were
outer-finalized. The sole development-closure attempt then terminated as
outer-finalized failure evidence.

Creation of
`bench/world_model_lifecycle/results/development/qualification-v1.7.0`
consumed the sole v1.7 development qualification. The prospectively sealed
single-use rule forbids rerunning the producer, audit, or closure and forbids
using the completed producer in a later protocol version.

No K3–K6 performance value in `result.json` or the accepted development audit
was opened, inspected, copied, summarized, compared, thresholded, selected on,
or used to choose the harness repair. The preserved v1.7 evidence does not
support a claim that Prospect learned, improved, retained an improvement, or
survived restart.

No v1.7 preformal authorization, formal binding, formal marker, formal
producer, formal audit, semantic review, or adjudication is permitted.

## Preserved producer and audit evidence

The canonical producer terminal is:

`bench/world_model_lifecycle/results/development/qualification-v1.7.0/producer-manifest.json`

Independent metadata and filesystem checks established:

- producer status: `completed`;
- lane: `development`;
- error: `null`;
- declared file count: 76;
- producer manifest SHA-256:
  `fe0d089ade9f4aef2e0b6b9f80ebfdb5304a232f82b2dac83b5e7376fadcce2b`;
- producer manifest bytes: 12,663;
- producer manifest link count: two; and
- deterministic same-inode outer completion:
  `bench/world_model_lifecycle/results/outer-completions/v1.7/06cb870ec26264fc9b1469c9969d31ed6eff96369db914a548a346fb566532a4.json`.

The producer manifest binds the opaque `result.json` member at 320,556,697
bytes with SHA-256
`af35b1d3c5bd028e56f0ca50b02f683853276b4ccc05680d904aeaf7b8e63cc2`.
The independent check hashed the raw file without decoding or inspecting its
metric values.

The finalized runtime seal has SHA-256
`ae85c7534d90218058eff813ea089b521efcb9a55ea3ec3f38fa9249154dabd5`,
1,913 bytes, link count two, and same-inode outer completion
`bench/world_model_lifecycle/results/outer-completions/v1.7/5a9096ec752ec478941985d44319a9c7a27661bea3391a33d26215e4244c7709.json`.

The accepted development-audit terminal is:

`bench/world_model_lifecycle/results/operator-v1.7/audits/development-audit-v1.7.0/operator-attempt.json`

Its authenticated non-performance metadata is:

- attempt kind: `audit`;
- lane: `development`;
- status: `accepted`;
- declared file count: 15;
- terminal SHA-256:
  `5182b70179fc9b936be2c9752c74a29c8c8ec3929363b4d791e797596abd4805`;
- terminal bytes: 3,273;
- terminal link count: two; and
- deterministic same-inode outer completion:
  `bench/world_model_lifecycle/results/outer-completions/v1.7/b502c4828a2aac49ebd9d18ea59a18b089a7665f9ab454840ef53efc91905105.json`.

## Preserved closure-failure evidence

The canonical closure terminal is:

`bench/world_model_lifecycle/results/operator-v1.7/closures/development-closure-v1.7.0/operator-attempt.json`

Strict verification established:

- attempt kind: `closure`;
- lane: `development`;
- status: `failure`;
- failure phase: `development_closure`;
- error type: `RuntimeError`;
- failure code: `runtime_error`;
- terminal SHA-256:
  `5b3ceaa1fa0c26bfa672e3bdf0f23baaac12dff78a68f33747233e0ae2a1aece`;
- terminal bytes: 5,266;
- terminal link count: two; and
- deterministic same-inode outer completion:
  `bench/world_model_lifecycle/results/outer-completions/v1.7/8265b6bd8417b4eac7d9fb4666e2a972a2baacde2425e46b3eee79bd66e99daf.json`.

The sole nonterminal member is `execution-failure.json`, with SHA-256
`231b6fc0d31059b884a944aad85eedfbb4164232d682919415b17040b95f23b5`,
252 bytes, and link count one. No development-closure marker or qualification
archive was published.

## Root cause

The closure constructor successfully verified the producer manifest, complete
result structure, exact two-seed matrix, accepted audit and sidecars, live
execution identity, clean source identity, captured runtime custody, and
prospective archive. Its final prepublication check then called
`_stable_regular_payload` for `result.json`. That helper has a default 64 MiB
limit and materializes the entire file, while the authenticated result is
320,556,697 bytes. It therefore deterministically raised:

```text
RuntimeError: development result exceeds its byte limit
```

This was independently reproduced under the exact v1.7 runtime executable,
45-row installed inventory, `-I -S -B` flags, seven-variable process
environment, and captured seal/bootstrap bytes, using only a disposable
noncanonical output that was removed. The diagnostic emitted no metric value
and modified no canonical evidence.

This is a closure-custody scaling defect. It is not evidence of a failure in
the world model, learning algorithm, optimizer, planner, controller, budgets,
controls, metrics, thresholds, or scientific gates.

The repair must stream the result through a no-follow descriptor, bind its
typed manifest size and digest, preserve exact link and before/after inode and
namespace identity, and test an artifact larger than 64 MiB plus mutation,
namespace-replacement, hard-link, and symlink cases. Merely increasing the
limit would preserve unnecessary memory scaling and defer the same defect.
The repair requires a new protocol version, fresh namespaces, fresh seeds,
fresh runtime custody, and a new prospective review.

## Confirmed absent authorization artifacts

The following canonical v1.7 authorization paths were absent when this review
was recorded:

- `bench/world_model_lifecycle/results/development/development-closure-v1.7.0.json`;
- `bench/world_model_lifecycle/results/development/preformal-test-report-v1.7.0.json`;
- `bench/world_model_lifecycle/results/operator-v1.7/bindings/formal-binding-v1.7.0`;
- `bench/world_model_lifecycle/results/formal/formal-launch-v1.7.0.json`; and
- any canonical v1.7 formal producer.

Their absence is required by the terminal disposition, not missing work that
may be completed under v1.7.
