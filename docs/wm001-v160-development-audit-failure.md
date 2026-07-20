# WM-001 v1.6.0 development-audit failure review

## Disposition

WM-001 protocol 1.6.0 is retired without a qualifying development closure and
without a formal attempt. The canonical two-seed development producer
completed and was outer-finalized, but its required independent audit failed
before producing a report. The failed audit attempt was also outer-finalized
as failure evidence.

Creation of
`bench/world_model_lifecycle/results/development/qualification-v1.6.0`
consumed the sole v1.6 development qualification. Under the prospectively
sealed single-use rule, a failure after that creation retires the version.
The preserved producer may not be re-audited to upgrade its disposition, and
it may not be resumed, overwritten, or replaced by a sibling attempt.

No K3–K6 performance value in `result.json` was opened, inspected, copied,
summarized, compared, thresholded, selected on, or used to change either the
scientific system or the next harness repair. The result is retained only as
an opaque, content-addressed member of the failed qualification record. It
does not support a claim that Prospect learned, improved, retained an
improvement, or survived restart.

No v1.6 development closure, preformal authorization, formal binding, formal
marker, formal producer, formal audit, semantic review, or adjudication is
permitted.

## Preserved producer evidence

The canonical producer terminal is:

`bench/world_model_lifecycle/results/development/qualification-v1.6.0/producer-manifest.json`

Independent metadata and filesystem checks established:

- producer status: `completed`;
- lane: `development`;
- error: `null`;
- declared file count: 76;
- producer manifest SHA-256:
  `aef90dcf7c36503b2b9265f1a6dc834aff7a558945c1124c5dfedc26239ce6e2`;
- producer manifest bytes: 12,663;
- producer manifest link count: two; and
- deterministic same-inode outer completion:
  `bench/world_model_lifecycle/results/outer-completions/v1.6/1fc3ced56c620d978839170ca93fd8a5d460a6c70e7de5dd3f49635dab1d2f0d.json`.

The producer manifest identifies the opaque result member as:

- path:
  `bench/world_model_lifecycle/results/development/qualification-v1.6.0/result.json`;
- SHA-256:
  `4aa844003116a5aa6284a086a32047291f3bb5edf7fc7ab541498e0687177a4d`;
- bytes: 320,602,497; and
- link count: one.

The independently computed file digest and byte count match the producer
manifest row. This check hashed the raw file without decoding or inspecting
its metric values.

The finalized runtime seal remains at
`bench/world_model_lifecycle/results/development/runtime-seal-v1.6.0.json`
with SHA-256
`1f2397721bcf2b52fdbe4d5ca41e0e8011fb7c428dfeffd2df55779b06f9c85a`,
1,913 bytes, and link count two. Its deterministic same-inode outer
completion is
`bench/world_model_lifecycle/results/outer-completions/v1.6/aabdb23228ab10161417982470811de1413525e4f25a93755e5dd2cad761af49.json`.

## Preserved audit-failure evidence

The canonical audit terminal is:

`bench/world_model_lifecycle/results/operator-v1.6/audits/development-audit-v1.6.0/operator-attempt.json`

Independent metadata and filesystem checks established:

- attempt kind: `audit`;
- lane: `development`;
- status: `failure`;
- error type: `AuditExecutionFailure`;
- failure code: `audit_execution_failure`;
- audit terminal SHA-256:
  `5100bab6ed9f3d3d2bb293ba4b2fd67e57c22b72755a5be9d3881f894e794b68`;
- audit terminal bytes: 2,063;
- audit terminal link count: two; and
- deterministic same-inode outer completion:
  `bench/world_model_lifecycle/results/outer-completions/v1.6/9f7317e27e32a170a5f7c0393f3d639df3d5699e36b6814c9cd5662f683de2d3.json`.

The terminal records no accepted audit execution, no independent-audit
report, and no reproduction receipt: `executions` is empty, `audit_file` is
`null`, `reproduction_file` is `null`, and
`reproduction_runtime_file` is `null`.

The preserved non-performance failure members are:

| Member | Bytes | SHA-256 |
| --- | ---: | --- |
| `execution-failure.json` | 265 | `8aa4b1e5bd5574ab6ebff39dc7a8b4d00b81ee48088e62b1b6748ffae70d8f04` |
| `audit-execution-01.failure.json` | 1,372 | `f7465f1ce31e76b205aa01f470f169c3b1a17c84c93a121f7ed33b9c10c54fb1` |
| `audit-execution-01.partial.runtime.json` | 1,896 | `5c0aedea3fe24956d2188d405b8edd7057f07644265b38ec1e07a39c371b746e` |
| `audit-execution-01.partial.invocation.json` | 320 | `11f21b957f59ad141e0c9df8070926fa603ee043a1703e71e770c962b90cb4fb` |
| `audit-execution-01.partial.stdout` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `audit-execution-01.partial.stderr` | 2,045 | `32fb1beb1b3ad3bba5e80070fda1e61005d8f3789cd861520a493b788c359e1b` |

Every member has link count one. The failure receipt records descriptor source
mode, child return code 1, and failure phase `report_validation`. Its recorded
support closure contains only:

- `protocol.json`, SHA-256
  `6f5c21d6e77683c283e09c6257c35abd0e6857e17620e585f414024852d972b2`;
  and
- `schemas/raw-result.schema.json`, SHA-256
  `1ce844aea5bc4167b6a151d5e694e356009d5b897639722bbb230541af0527bb`.

## Root cause

The independent auditor runs from a private capture directory. Consequently,
its `HERE = Path(__file__).resolve().parent` denotes that private directory,
not the installed package directory. Restart-runtime validation tried to hash
the sibling path
`/tmp/prospect-audit-runner-c13_xql3/capture/producer_bootstrap.py`, but
`producer_bootstrap.py` was absent from the explicitly captured support
closure. The resulting `FileNotFoundError` escaped before the auditor could
emit canonical JSON; this agrees with the empty captured stdout, return code
1, and `report_validation` failure.

This is an outcome-auditor capture-closure defect. It is not evidence of a
failure in the world model, learning algorithm, optimizer, planner,
controller, budgets, controls, metrics, thresholds, or scientific gates.

The harness repair must make every source file dereferenced relative to the
captured auditor an explicit, content-bound support file, propagate that
support identity through development and sealed formal outcome-audit
contracts, and test replay from fresh private directories. That repair
requires a new protocol version, fresh canonical namespaces, and fresh seeds;
it cannot retroactively qualify v1.6.

## Confirmed absent authorization artifacts

The following canonical v1.6 authorization paths were absent when this review
was recorded:

- `bench/world_model_lifecycle/results/development/development-closure-v1.6.0.json`;
- `bench/world_model_lifecycle/results/operator-v1.6/closures/development-closure-v1.6.0`;
- `bench/world_model_lifecycle/results/operator-v1.6/bindings/formal-binding-v1.6.0`;
- `bench/world_model_lifecycle/results/formal/formal-launch-v1.6.0.json`; and
- any canonical v1.6 formal producer.

Their absence is required by the terminal disposition, not missing work that
may be completed under v1.6.
