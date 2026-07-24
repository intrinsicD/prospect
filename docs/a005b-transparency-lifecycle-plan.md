# A-005b transparency and signing-key lifecycle plan

Status: queued; planning only.

## Dependency

Begin after A-005a freezes canonical signed-envelope and offline-verification
vectors and after an independently operated transparency or signing provider is
selected prospectively. Stable evidence schemas alone are insufficient.

## Falsifiable objective

An independent verifier can use externally retrievable receipts and a declared
key lifecycle to detect package rollback, same-sequence equivocation,
revocation violations, and invalid key rotation without trusting the producing
repository.

## Gates

1. Freeze provider identity/API, sequence or epoch rule, append-only receipt or
   inclusion-proof format, timestamp semantics, key custody, rotation,
   revocation, compromise, and outage policy.
2. Bind each A-005a envelope digest and predecessor receipt into one externally
   ordered identity before publication.
3. From a clean machine, retrieve or verify the external receipt and reject an
   older accepted package presented as current.
4. Detect conflicting packages at one sequence, omitted predecessor links,
   invalid rotations, use after revocation, and signatures outside declared
   validity intervals.
5. Preserve negative dispositions and compromise notices so a replacement key
   cannot rewrite historical evidence.
6. Exercise provider outage and retention expiry under a prospectively declared
   fail-closed or degraded-verification policy.

## Exclusions

No scientific-correctness, compromised verifier trust-root, hostile kernel,
hardware, anonymity, or indefinite provider-availability claim. This task does
not alter the signed scientific payload defined by A-005a.

## Intended repository surface

Provider-neutral receipt schemas and verification adapters under
`bench/assurance/attestation/`, adversarial tests in
`tests/test_a005b_transparency.py`, and no private signing credentials in the
repository.

## Next action

Compare independently operated transparency/signing options for append-only
proofs, offline verification, key custody, revocation, retention, API
stability, and cost. Freeze a provider only after A-005a golden vectors pass.
