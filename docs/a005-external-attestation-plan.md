# A-005a canonical signed-evidence-envelope plan

Status: queued; planning only.

## Dependency

Begin after the canonical evidence-envelope schema and its versioning rule are
stable. This task freezes offline envelope construction and verification. It
does not claim rollback/equivocation resistance from a transparency service;
that belongs to A-005b.

## Falsifiable objective

An offline verifier, without trusting the producing repository, can bind a
Prospect evidence package to its implementation and execution identities and
detect package mutation, member substitution, cross-run replay, and invalid
signatures under a prospectively supplied public trust root.

## Gates

1. Declare trusted and untrusted actors, signature algorithm, canonical
   payload, package/run identity, trust-root input, and verification policy.
2. Bind the complete package manifest, source/binding identity, execution
   environment, result, audit, semantic review, adjudication, and predecessor
   attestation.
3. Verify without network access from a clean checkout and externally supplied
   package, signature, and public trust root.
4. Reject bit mutation, omitted/extra members, path substitution, replay into a
   new run, an unknown signing identity, and an invalid signature.
5. Verify from a clean machine using only the public trust root and published
   package.
6. Emit a canonical verification receipt that A-005b can bind into an external
   sequence without reinterpreting scientific results.

## Exclusions

No rollback, equivocation, revocation, rotation, timestamp, compromised-key,
verifier-trust-root, kernel, or hardware claim. Those lifecycle claims require
A-005b or a later hardware-backed protocol. Signatures establish integrity and
provenance, not scientific correctness.

## Intended repository surface

An implementation-independent schema and verifier under
`bench/assurance/attestation/`, adversarial tests in
`tests/test_a005a_envelope.py`, and no private signing material in the
repository.

## Next action

Write the offline threat model, canonical envelope schema, and golden
verification vectors. Provider comparison and key lifecycle design start only
in A-005b.
