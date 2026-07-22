.RECIPEPREFIX = >
.PHONY: install install-runtime test test-runtime lint fmt typecheck typecheck-runtime epistemic-diagnostics epistemic-gate wm001-development check check-runtime tree

install:
> python -m pip install -e ".[dev]"

install-runtime:
> python -m pip install -e ".[dev,runtime]"

test:
> pytest -q tests/test_epistemic_*.py

test-runtime:
> pytest -q tests/test_world_model_*.py

lint:
> ruff check src/prospect bench tests

fmt:
> ruff format src/prospect bench tests

typecheck:
> mypy

typecheck-runtime:
> mypy --follow-imports=skip \
>   bench/world_model_lifecycle/audit_runner.py \
>   bench/world_model_lifecycle/artifact.py \
>   bench/world_model_lifecycle/artifact_audit.py \
>   bench/world_model_lifecycle/adjudication.py \
>   bench/world_model_lifecycle/binding.py \
>   bench/world_model_lifecycle/experiment.py \
>   bench/world_model_lifecycle/launch_bootstrap.py \
>   bench/world_model_lifecycle/operator.py \
>   bench/world_model_lifecycle/preformal.py \
>   bench/world_model_lifecycle/producer_bootstrap.py \
>   bench/world_model_lifecycle/rehearsal.py \
>   bench/world_model_lifecycle/restore_eval.py \
>   bench/world_model_lifecycle/run.py

epistemic-diagnostics:
> PYTHONPATH=src python -m bench.epistemic.run_maturity --diagnostics

epistemic-gate:
> PYTHONPATH=src python -m bench.epistemic.run_maturity

wm001-development:
> @echo "Direct WM-001 entry is disabled; use docs/wm001-v1170-operator-runbook.md" >&2
> @exit 2

check: lint typecheck test epistemic-diagnostics

check-runtime: lint typecheck typecheck-runtime test test-runtime epistemic-diagnostics

tree:
> find . -not -path '*/.*' -not -path '*/__pycache__/*' -type f | sort
