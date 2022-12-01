PYTEST_ARGS?=
SHELL = /bin/sh

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

coverage: ## check code coverage quickly with the default Python
	pytest --cov=fv3gfs --cov-report=html
	$(BROWSER) htmlcov/index.html

test: ## run tests quickly with the default Python
	pytest $(PYTEST_ARGS) tests

test_mpi:
	mpirun -n 6 --allow-run-as-root --mca btl_vader_single_copy_mechanism none --oversubscribe pytest $(PYTEST_ARGS) tests/mpi
	$(MAKE) -C examples/mpi

lint:
	black --diff --check $(PYTHON_FILES) $(PYTHON_INIT_FILES)
	flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	flake8 --ignore=F401 $(PYTHON_INIT_FILES)
	mypy --follow-imports silent --show-error-codes fv3gfs
	@echo "LINTING SUCCESSFUL"

clean: clean-build clean-pyc clean-test
	$(MAKE) -C examples/mpi clean

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

reformat:
	black $(PYTHON_FILES) $(PYTHON_INIT_FILES)

dist: clean ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel
	ls -l dist

release: dist ## package and upload a release
	twine upload dist/*

.PHONY: clean test lint reformat test_mpi coverage help
