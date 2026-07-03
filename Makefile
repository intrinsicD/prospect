.RECIPEPREFIX = >
.PHONY: install test lint fmt gate gate-all tree

install:
> pip install -e ".[dev]" --break-system-packages

test:
> pytest -q

lint:
> ruff check .

fmt:
> ruff format .

# usage: make gate PHASE=P1
gate:
> python -m bench $(PHASE)

# regression ratchet (P0-007): re-run every shipped phase's gate; fail on regression
gate-all:
> python -m bench --all

tree:
> find . -not -path '*/.*' -not -path '*/__pycache__/*' -type f | sort
