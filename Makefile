.RECIPEPREFIX = >
.PHONY: install test lint fmt gate tree

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

tree:
> find . -not -path '*/.*' -not -path '*/__pycache__/*' -type f | sort
