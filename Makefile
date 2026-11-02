.PHONY: install lint typecheck test test-fast style smoke agent-check ci

install:
	pip install -r requirements.txt
	pip install -e .

smoke:
	python -m sevs.cli smoke-test --config configs/eval/eval_default.yaml

lint:
	ruff check src tests scripts

typecheck:
	mypy src

test:
	python -m pytest -q

test-fast:
	python -m pytest -x -q -m "not slow"

style:
	python3 scripts/agent/check_style.py --changed .

# Fast checks the Stop hook can run. Add lint and typecheck once each passes.
agent-check: test-fast style

ci: lint typecheck test style
	python3 scripts/agent/check_tests.py --warn
