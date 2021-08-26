include docker/Makefile.image_names

GCR_URL = us.gcr.io/vcm-ml
REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_standard
FORTRAN_SERIALIZED_DATA_VERSION=7.2.5
WRAPPER_IMAGE = us.gcr.io/vcm-ml/fv3gfs-wrapper:gnu9-mpich314-nocuda
DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=n
NUM_RANKS ?=6
VOLUMES ?=
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
TEST_ARGS ?=
TEST_DATA_HOST ?=$(CWD)/test_data/$(EXPERIMENT)
FV3UTIL_DIR=$(CWD)/fv3gfs-util

FV3=fv3core
FV3_PATH ?=/$(FV3)
PHY=fv3gfs-physics
PHY_PATH ?=/$(PHY)

TEST_DATA_RUN_LOC ?=/test_data
PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')
TEST_DATA_TARFILE=dat_files.tar.gz
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)
CORE_TAR=$(SARUS_FV3CORE_IMAGE).tar
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS)
BASE_INSTALL?=$(FV3)-install-serialbox
DATA_BUCKET= $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/

TEST_TYPE=$(word 3, $(subst _, ,$(EXPERIMENT)))
THRESH_ARGS=--threshold_overrides_file=$(FV3_PATH)/tests/savepoint/translate/overrides/$(TEST_TYPE).yaml
PYTEST_MAIN=pytest $(TEST_ARGS) $(FV3_PATH)/tests/main
PYTEST_SEQUENTIAL=pytest --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) $(FV3_PATH)/tests/savepoint
# we can't rule out a deadlock if one test fails, so we must set maxfail=1 for parallel tests
PYTEST_PARALLEL=$(MPIRUN_CALL) python -m mpi4py -m pytest --maxfail=1 --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) -m parallel $(FV3_PATH)/tests/savepoint
ifeq ($(DEV),y)
	VOLUMES += -v $(CWD)/$(FV3):/$(FV3) -v $(CWD)/$(PHY):/$(PHY) -v $(CWD)/tests:/$(FV3)/tests -v $(FV3UTIL_DIR):/usr/src/fv3gfs-util
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(CUDA_FLAGS) $(FV3CORE_IMAGE)

update_submodules_base:
	if [ ! -f $(FV3UTIL_DIR)/requirements.txt  ]; then \
		git submodule update --init fv3gfs-util ; \
	fi
	if [ ! -f $(FV3_PATH)/requirements.txt  ]; then \
		git submodule update --init fv3core ; \
	fi
	if [ ! -f $(PHY_PATH)/requirements.txt  ]; then \
		git submodule update --init fv3gfs-physics ; \
	fi

build:
	if [ $(PULL) == True ]; then \
		$(MAKE) pull_environment_if_needed; \
	else \
		$(MAKE) build_environment; \
	fi
	$(MAKE) -C docker fv3gfs_image

build_environment:
	$(MAKE) -C docker build_core_deps

pull_environment_if_needed:
	$(MAKE) -C docker pull_core_deps_if_needed

dev:
	docker run --rm -it \
		--network host \
		-v $(TEST_DATA_HOST):$(TEST_DATA_RUN_LOC) \
		-v $(CWD):/port_dev \
		$(FV3GFS_IMAGE) bash

test_util:
	$(MAKE) -C fv3gfs-util test
	
savepoint_tests:
	$(MAKE) -C fv3core savepoint_tests

savepoint_tests_mpi:
	$(MAKE) -C fv3core savepoint_tests_mpi

constraints.txt: requirements.txt requirements/requirements_wrapper.txt requirements/requirements_lint.txt
	pip-compile $^ --output-file constraints.txt