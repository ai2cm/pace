SHELL := /bin/bash
include docker/Makefile.image_names
include Makefile.data_download
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

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
CHECK_CHANGED_SCRIPT=$(CWD)/changed_from_main.py

VOLUMES ?=
ifeq ($(DEV),y)
	VOLUMES += -v $(ROOT_DIR):/pace
endif

### testing variables

FV3=fv3core
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(PACE_IMAGE)
ifeq ($(CONTAINER_CMD),)
	PACE_PATH?=$(ROOT_DIR)
else
	PACE_PATH?=/pace
endif
ifeq ($(CONTAINER_CMD),)
	TEST_DATA_RUN_LOC=$(TEST_DATA_HOST)
else
	TEST_DATA_RUN_LOC=$(PACE_PATH)/test_data/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/$(TARGET)
endif
NUM_RANKS ?=6
MPIRUN_ARGS ?=--oversubscribe
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS) $(MPIRUN_ARGS)
TEST_ARGS ?=-v
TEST_TYPE=$(word 3, $(subst _, ,$(EXPERIMENT)))
THRESH_ARGS=--threshold_overrides_file=$(PACE_PATH)/fv3core/tests/savepoint/translate/overrides/$(TEST_TYPE).yaml

###


build:
ifeq ($(DEV),n)
	DOCKER_BUILDKIT=1 docker build \
		-f $(CWD)/Dockerfile \
		-t $(PACE_IMAGE) \
		.
endif

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

savepoint_tests: get_test_data build
	$(CONTAINER_CMD) bash -c "pip3 list && cd $(PACE_PATH) && pytest --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) $(PACE_PATH)/fv3core/tests/savepoint"

savepoint_tests_mpi: get_test_data build
	$(CONTAINER_CMD) bash -c "pip3 list && cd $(PACE_PATH) && $(MPIRUN_CALL) python3 -m mpi4py -m pytest --maxfail=1 --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) -m parallel $(PACE_PATH)/fv3core/tests/savepoint"

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
