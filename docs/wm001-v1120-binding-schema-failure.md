# WM-001 v1.12 binding-schema failure

Date: 2026-07-21

Disposition: **retired — failed binding, no formal authority**

## Terminal outcome

The exact WM-001 protocol 1.12.0 harness was sealed at Git commit
`d7eca5c0cd75802e73ec7a0cd3436ec7b2ecdd5f`. Its protocol, independent
prospective review, 666-test suite, lint, two type-check scopes, installed
source snapshots, exact 45-row runtime lock, unequal QA/runtime environments,
warning-as-error PyTorch precision probe, and CUDA gate passed.

The prospective runtime seal was created and outer-finalized. The exact
result-free command-10 rehearsal returned zero, emitted canonical passing JSON,
and wrote exactly zero stderr bytes. Its separately retained capture remained
result-free. The four real-subprocess audit modules then passed all 193 tests.

The sole development producer completed both fresh v1.12 masters and was
outer-finalized. Its independent development audit was accepted, reproduced
byte-for-byte, and outer-finalized. The sole closure transaction published the
canonical closure, retained its fresh sealed-runtime reopen, and outer-finalized
the accepted attempt. The required post-finalization sealed-runtime reopen
passed.

The one-shot preformal generator then completed all ten fixed commands and
published the complete 20-log report. Both the separate QA-side report
verifier and the recorded-runtime verifier accepted it. In particular, every
captured command stderr was the exact zero-byte stream required by the
protocol.

The sole binding transaction copied and authenticated the accepted evidence,
constructed its formal binding, and then failed JSON Schema validation at:

```text
source.test_log_files.1.bytes: 0 is less than the minimum of 1
```

The transaction published and outer-finalized a terminal binding attempt with
status `failure`, error type `Violation`, and failure code `violation`.
Protocol 1.12.0 may not be repaired, retried, resumed, rebound, or launched
formally. No v1.12 formal launch marker, formal producer, formal audit,
semantic review, adjudication marker, or adjudication package exists.

## Evidence identity

- sealed Git commit:
  `d7eca5c0cd75802e73ec7a0cd3436ec7b2ecdd5f`
- protocol SHA-256:
  `d64aede84e402d05bd587e1fdf2694381ab6742a28ca19ed88097d0480fa5b80`
- prospective-review SHA-256:
  `07a6c317df8c95dc0f2aaa55e3e803f257a907fcea047140b766c016182410e4`
- prospective implementation-manifest SHA-256:
  `af6a87f5f2a8b3e3e5e30ed660bdfabf096a7a92e8a6c6926b81dbf9ed47988c`
- runtime-seal SHA-256:
  `9a2af4f994e6bc4c1dbe154927314078ecb78abb65f9fc5246883ea9196a38da`
- producer-manifest SHA-256:
  `fb8a5f1cdf55e4b73b5f202181e110be34034c3f63454e4ef7f2a3f5b28c36c9`
- raw-result SHA-256, emitted by the passing sealed closure reopen:
  `c599deca64a527e8555cc8b0be483336a5d6270c09f1107b09cd168c7b2d4a35`
- accepted development-audit terminal SHA-256:
  `3705c1075ba8029517b16f8ec51d86295c28343bb4523cfb5065241cc66dbcb9`
- canonical development-closure SHA-256:
  `0dbe39f8501b2ae66083cf96c274bda9166973b012c2b84de3f10e06fb70c998`
- accepted closure-attempt terminal SHA-256:
  `5046b84f6cfe3a10e7bd575ffe99fdfd553c7399f1825a12feac8fa923205bf9`
- accepted preformal report SHA-256:
  `cb3eda99f4d5659cef03d62a22466a816859ed85ebc3bba09063144d3d84844e`
- terminal formal-binding payload SHA-256:
  `557901e18878928190f5e33c6575b597a26696dc131b71c48d8af3f2d1604925`
- binding execution-failure SHA-256:
  `465cc99a34d42f47d86adecff8c9332bf34dfeace1e26ac12a3a24bc8509a9e7`
- binding error message: 101 bytes with SHA-256
  `157283ad48452f7a64abc08fd45090b5a7793542c3bf7fdc6c9263b3ed0b0f20`
- failed, outer-finalized binding-attempt terminal SHA-256:
  `12df27e0a3f65273ddd778d64b652f803611ea7dccb7bef5affe932648637d12`
  (the terminal and deterministic outer marker are inode `71996576` with
  link count two)
- recursive fixed-root evidence inventory
  (`qualification-v1.12.0`, runtime seal, development closure, preformal
  bundle, `operator-v1.12`, and `outer-completions/v1.12`): 173 sorted rows,
  comprising 10 directories and 163 regular files
  (1,120,084,598 regular-file bytes); compact key-sorted canonical row JSON is
  53,107 bytes with SHA-256
  `cc36413bd1ddf6659c6a67ca3771bb6e3e182a12332087773ece27cea4eaad86`.
- formal launch marker: absent
- formal audit marker: absent
- semantic review: absent
- adjudication marker and package: absent

No performance or K value was opened, printed, summarized, compared, or used
to choose the repair.

## Exact cause

`schemas/formal-binding.schema.json` reused one `fileDigest` definition for two
semantically different classes:

1. implementation and evidence files that must be nonempty; and
2. preformal stdout/stderr log files, where a zero-byte stderr is both valid
   and required for successful command rows.

That shared definition set `bytes.minimum` to `1`. Binding construction
correctly preserved each log's actual byte count, including zero-byte stderr,
but the terminal schema pass therefore rejected the first empty log row. All
ten odd-numbered log-manifest rows were required zero-byte stderr streams and
would violate the same shared definition. The preformal report schema and
semantic verifiers had already accepted those correct empty streams; the
mismatch existed only between binding generation semantics and the
formal-binding JSON Schema.

Existing tests checked many binding mutations but did not schema-validate a
complete, legitimate 20-row log manifest containing the required empty stderr
rows. Synthetic binding fixtures commonly used nonempty placeholder payloads,
so the schema/producer contradiction escaped the prospective gate.

During post-closure diagnosis, an additional read-only
`verify_operator_attempt` call was mistakenly made from the intentionally
larger QA environment. It correctly refused the eleven QA-only distributions.
It created or changed no evidence path, and the subsequently launched exact
sealed-runtime closure reopen passed. This operator-side diagnostic was not
the binding failure and supplied no authority to repeat any lifecycle step.

## Required fresh-version repair

A successor protocol must:

1. introduce distinct schema definitions for nonempty files and log files
   whose payload may legitimately be empty, without weakening nonempty
   implementation, report, conformance, or source evidence;
2. bump the formal-binding representation because its accepted value set
   changes, while leaving unrelated representations unchanged;
3. schema-validate a complete real-shaped binding fixture with ten stdout and
   ten zero-byte stderr rows, and prove that negative byte counts, wrong empty
   digests, missing rows, reordered rows, and nonempty command stderr remain
   rejected by the appropriate schema or semantic verifier;
4. add a producer-to-schema integration test that constructs the same binding
   object written by the one-shot binding transaction and validates it before
   any lifecycle evidence is created;
5. audit every `minimum`, `minLength`, and shared digest definition across the
   raw-result and formal-binding schemas against producer and verifier
   semantics;
6. use fresh versioned paths, seeds, environments, wheel, lock, prospective
   review, seal, development evidence, binding, and formal authority; and
7. preserve every v1.12 producer, audit, closure, preformal, failed-binding,
   execution-failure, and outer-completion byte unchanged.
