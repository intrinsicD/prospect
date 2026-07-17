.RECIPEPREFIX = >
.PHONY: install test lint fmt typecheck gate gate-all bench-hard bench-multimodal-smoke bench-multimodal bench-multimodal-verify tree

install:
> pip install -e ".[dev,learn]" --break-system-packages

test:
> pytest -q

lint:
> ruff check .

fmt:
> ruff format .

# full type hints are enforced (P0-009); protocol conformance is checked statically
typecheck:
> mypy

# usage: make gate PHASE=P1
gate:
> python -m bench $(PHASE)

# regression ratchet (P0-007): re-run every shipped phase's gate; fail on regression
gate-all:
> python -m bench --all

# optional harder-benchmark probe (BH-001, ADR-0011) — NON-gated, needs the
# `[bench-hard]` extra (`pip install -e '.[bench-hard]'`). Never part of gate-all/CI.
bench-hard:
> python -m bench.hard

# optional MM-001 real-media systems preflight; requires `[bench-multimodal]`
bench-multimodal-smoke:
> python -m bench.multimodal_preflight smoke

bench-multimodal:
> python -m bench.multimodal_preflight run

bench-multimodal-verify:
> python -m bench.multimodal_preflight verify-semantic

tree:
> find . -not -path '*/.*' -not -path '*/__pycache__/*' -type f | sort
