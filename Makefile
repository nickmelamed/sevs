.PHONY: install lint typecheck test test-fast style smoke agent-check ci

# Use the repo's virtualenv when it exists, so the checks do not depend on the shell's PATH.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e ".[dev]"

smoke:
	$(PY) -m sevs.cli smoke-test --config configs/eval/eval_default.yaml

lint:
	$(PY) -m ruff check src tests scripts

typecheck:
	$(PY) -m mypy src

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -x -q -m "not slow"

style:
	python3 scripts/agent/check_style.py --changed .

# Fast checks the Stop hook runs.
agent-check: lint typecheck test-fast style

ci: lint typecheck test style
	python3 scripts/agent/check_tests.py --warn
