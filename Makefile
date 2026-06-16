.PHONY: install lint format typecheck test docs FORCE

install: FORCE
	pip install -e .[dev,docs]

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

docs: FORCE
	sphinx-build -W --keep-going -b html docs/source docs/build/html

FORCE:
