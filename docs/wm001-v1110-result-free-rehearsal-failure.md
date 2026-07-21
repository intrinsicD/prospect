# WM-001 v1.11 result-free rehearsal failure

Status: terminal engineering record. Protocol 1.11.0 is retired without a
development qualification, preformal report, binding, formal launch, or
performance claim.

## What completed

The exact prospective harness was sealed at Git commit
`2924a617c74abc520d315189769e19e2d5f25834`. Its protocol, review, 660-test
suite, lint, type checks, installed-source snapshots, 45-row runtime lock,
unequal QA/runtime environments, and CUDA gate passed.

The prospective runtime seal was then created at its canonical v1.11 path and
outer-finalized through a same-inode two-link completion. The sealed,
result-free `bootstrap-inventory-conformance` rehearsal completed its fresh
interpreter, private-path and descriptor, restart-runtime, inventory, and
matrix-contract checks successfully.

## Why v1.11 is retired

The successful rehearsal emitted a benign PyTorch 2.9 `UserWarning` on stderr
when active code accessed the deprecated `allow_tf32` interface. The warning
revealed an exact-contract incompatibility: preformal command 10 requires
zero-byte stderr, so the later one-shot report could not have passed even
though the rehearsal's semantic object was valid.

Changing the TF32 calls, runtime-identity fields, tests, or installed wheel
after the prospective seal would violate the frozen-byte rule. Protocol 1.11
is therefore terminal. Its seal, outer completion, runtime lock, environments,
wheel, and console record must not be removed, overwritten, resumed, or reused
as v1.12 evidence.

## Outcome boundary

No canonical v1.11 development producer root was created. No experience was
collected, no model was trained, no development or formal metric was produced,
and no prior retained K or performance value was opened.

## Required successor repair

Protocol 1.12 must keep the zero-stderr contract and unchanged scientific
system while:

1. replacing every legacy TF32 getter/setter with PyTorch 2.9's explicit
   `fp32_precision` hierarchy;
2. binding the global, backend, convolution, RNN, and matmul precision strings
   in a new runtime-identity request/report schema;
3. requiring zero stderr inside repeated restart-runtime and prebinding
   receipts, not only at preformal command 10's outer boundary;
4. adding static and fresh-subprocess tests that make any legacy TF32 access
   fatal; and
5. using fresh v1.12 seeds, namespaces, environments, wheel, lock, prospective
   review, seal, and one-shot lifecycle.
