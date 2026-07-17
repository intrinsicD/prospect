.RECIPEPREFIX = >
.PHONY: install install-runtime test test-runtime lint fmt typecheck epistemic-diagnostics epistemic-gate check tree

install:
> python -m pip install -e ".[dev]"

install-runtime:
> python -m pip install -e ".[dev,runtime]"

test:
> pytest -q tests/test_epistemic_*.py

test-runtime:
> pytest -q tests/test_epistemic_storage.py

lint:
> ruff check src/prospect bench/epistemic tests/test_epistemic_*.py

fmt:
> ruff format src/prospect bench/epistemic tests/test_epistemic_*.py

typecheck:
> mypy

epistemic-diagnostics:
> PYTHONPATH=src python -m bench.epistemic.run_maturity --diagnostics

epistemic-gate:
> PYTHONPATH=src python -m bench.epistemic.run_maturity

check: lint typecheck test epistemic-diagnostics

tree:
> find . -not -path '*/.*' -not -path '*/__pycache__/*' -type f | sort
