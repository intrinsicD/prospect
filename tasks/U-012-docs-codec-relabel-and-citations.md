# U-012 — Docs: correct the codec label + add rejected-alternative / validation citations

- **Status:** ready
- **Phase:** U (upgrade track; docs-only)
- **Requirements:** R6 (codec), plus ADR provenance across the board
- **ADRs:** ADR-0001 (codec), ADR-0002 (EDL rejected), ADR-0004 (CoALA/CaMeL),
  ADR-0009/0010/0012 (validation citations)
- **Depends on:** none (docs task; can run in parallel with anything)
- **Phase gate:** none (documentation) — `make test`/`lint`/`typecheck` stay clean
- **Source:** `docs/sota-review-2026-07.md` U-012 · [BCT](https://arxiv.org/abs/1912.03373)
  · [ImageBind](https://arxiv.org/abs/2305.05665) · [EDL "Mirage"](https://arxiv.org/abs/2402.06160)

## Goal
Fix documentation that is now inaccurate or under-cited, per the review:
1. `codec.py` is labelled "Perceiver-IO-style" but has no cross-attention, latent array,
   or query-based decoding — it is per-modality adapters aligned into a shared incumbent
   space (ImageBind-style) + textbook backward-compatible training (BCT) migration.
   Relabel in the module docstring and architecture.md.
2. Add the rejected-alternative and validation citations the review surfaced so the ADRs
   carry their literature anchors (they currently state mechanisms without them).

## Non-goals
- **No code behaviour change** — this is docs-sync only (the codec's design is correct at
  its scale; only its *description* is wrong).
- Not restating the whole review in the ADRs — one-line citations at the right decisions.

## Interface to satisfy
Documentation only: `src/prospect/codec.py` docstring (codec.py:1-39), `docs/architecture.md`,
and the relevant ADRs. No `interfaces.py`/`types.py` change.

## Approach (brief)
- codec.py + architecture.md glossary/codec bullet: "Perceiver-IO-style multi-modality
  codec" → "adapter-alignment into a shared incumbent latent (ImageBind-style) with
  distill-first backward-compatible migration (BCT)". Keep the true-cross-attention item
  as the earned-later future seam it already is.
- ADR-0001/roadmap P6: cite BCT (the migration *is* backward-compatible training) and its
  documented quality ceiling as the reason the retrain-fallback exists.
- ADR-0002: cite the EDL "Mirage" result (NeurIPS 2024) as the rejected alternative for
  the epistemic/aleatoric split; DDU/SNGP as the basis for distance-aware epistemic.
- ADR-0004: cite CoALA (retrieval-as-action) and CaMeL (structural provenance) as
  convergent validation.
- ADR-0009/0010/0012: cite V-JEPA 2 (frozen-embedding control), Garrido et al. 2026 /
  CLAM (continuous latent actions beat VQ), BCO/UniPi (IDM action recovery) as validation.

## Acceptance criteria
- [ ] "Perceiver-IO-style" removed from codec.py and architecture.md; replaced with the
      accurate ImageBind/BCT description.
- [ ] Each named ADR carries its rejected-alternative or validation citation (linking to
      `docs/sota-review-2026-07.md` for the full context).
- [ ] `make test` green, `make lint` clean, `make typecheck` clean (no code change, but
      the ratchet stays green).

## Test plan
- No new tests (docs). Verify `make gate-all` still green (unchanged behaviour) and links
  resolve.

## Docs-sync checklist
- [ ] Status → done.
- [ ] codec.py docstring corrected; architecture.md codec bullet + glossary corrected.
- [ ] ADR-0001/0002/0004/0009/0010/0012 citations added.
- [ ] `docs/sota-review-2026-07.md`: mark U-012 shipped.

## Gate result
Docs-only — no gate. Record the commit that lands the relabel + citations.
