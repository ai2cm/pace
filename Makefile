SHELL := /bin/bash
include docker/Makefile.image_names

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=n
RUN_FLAGS ?=--rm
CHECK_CHANGED_SCRIPT=$(CWD)/changed_from_main.py

ifeq ($(DEV),y)
	VOLUMES += -v $(CWD):/pace
endif

build:
	DOCKER_BUILDKIT=1 docker build \
		-f $(CWD)/Dockerfile \
		-t $(PACE_IMAGE) \
		.

enter:
	docker run --rm -it \
		--network host \
		$(VOLUMES) \
	$(PACE_IMAGE) bash

dev:
	DEV=y $(MAKE) enter

test_util:
	if [ $(shell $(CHECK_CHANGED_SCRIPT) pace-util) != false ]; then \
		$(MAKE) -C pace-util test; \
	fi

savepoint_tests:
	$(MAKE) -C fv3core $@

savepoint_tests_mpi:
	$(MAKE) -C fv3core $@

dependencies.svg: dependencies.dot
	dot -Tsvg $< -o $@

constraints.txt: dsl/requirements.txt fv3core/requirements.txt pace-util/requirements.txt fv3gfs-physics/requirements.txt driver/requirements.txt requirements_docs.txt requirements_lint.txt external/gt4py/setup.cfg requirements_dev.txt
	pip-compile $^ --output-file constraints.txt
	sed -i.bak '/\@ git+https/d' constraints.txt
	rm -f constraints.txt.bak

physics_savepoint_tests:
	$(MAKE) -C fv3gfs-physics $@

physics_savepoint_tests_mpi:
	$(MAKE) -C fv3gfs-physics $@

update_submodules_venv:
	if [ ! -f $(CWD)/external/daint_venv/install.sh  ]; then \
                git submodule update --init external/daint_venv; \
        fi

test_driver:
	DEV=$(DEV) $(MAKE) -C driver test

driver_savepoint_tests_mpi:
	DEV=$(DEV) $(MAKE) -C  fv3gfs-physics $@

docs: ## generate Sphinx HTML documentation
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

lint:
	pre-commit run --all-files

.PHONY: docs servedocs
