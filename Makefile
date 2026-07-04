.RECIPEPREFIX = >
.PHONY: install test lint fmt typecheck gate gate-all tree

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

tree:
> find . -not -path '*/.*' -not -path '*/__pycache__/*' -type f | sort
