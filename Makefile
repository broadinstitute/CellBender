.PHONY: install lint format typecheck test FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall cellbender

lint: FORCE
	ruff check .
	ruff format --check .

format: FORCE
	ruff check --fix .
	ruff format .

typecheck: FORCE
	mypy cellbender tests

test: FORCE
	pytest -v tests/

FORCE:
