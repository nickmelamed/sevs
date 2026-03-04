.PHONY: install lint test smoke

install:
	pip install -r requirements.txt
	pip install -e .

smoke:
	python -m sevs.cli smoke-test --config configs/eval/eval_default.yaml

test:
	python -m pytest -q
