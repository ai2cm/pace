include ../docker/Makefile.image_names

GCR_URL = us.gcr.io/vcm-ml
EXPERIMENT ?=c12_6ranks_standard
SHELL=/bin/bash
CWD=$(shell pwd)
DEV ?=y
NUM_RANKS ?=6
VOLUMES ?=
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
RUN_FLAGS += -e FV3_DACEMODE=$(FV3_DACEMODE)
TEST_ARGS ?=
TARGET ?=dycore
FV3UTIL_DIR=$(CWD)/external/util
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
TEST_DATA_ROOT ?=$(ROOT_DIR)/test_data/
TEST_DATA_FTP ?=in/put/abc/cosmo/fuo/pace/fv3core/

FV3=fv3core
FV3_PATH ?=/pace/$(FV3)

ifeq ($(DEV),y)
	VOLUMES += -v $(CWD)/..:/pace
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(PACE_IMAGE)

ifneq (,$(findstring $(PACE_IMAGE),$(CONTAINER_CMD)))
	TEST_DATA_RUN_LOC =/test_data
else
	TEST_DATA_RUN_LOC=$(TEST_DATA_HOST)
endif

PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')
TEST_DATA_TARFILE=dat_files.tar.gz
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)
MPIRUN_ARGS ?=--oversubscribe
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS) $(MPIRUN_ARGS)
BASE_INSTALL?=$(FV3)-install-serialbox
DATA_BUCKET= $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/$(TARGET)/

TEST_TYPE=$(word 3, $(subst _, ,$(EXPERIMENT)))
THRESH_ARGS=--threshold_overrides_file=$(FV3_PATH)/tests/savepoint/translate/overrides/$(TEST_TYPE).yaml
PYTEST_MAIN=pytest $(TEST_ARGS) $(FV3_PATH)/tests/main
PYTEST_SEQUENTIAL=pytest --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) $(FV3_PATH)/tests/savepoint
# we can't rule out a deadlock if one test fails, so we must set maxfail=1 for parallel tests
PYTEST_PARALLEL=$(MPIRUN_CALL) python3 -m mpi4py -m pytest --maxfail=1 --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) -m parallel $(FV3_PATH)/tests/savepoint
ifeq ($(DEV),y)
	VOLUMES += -v $(CWD)/$(FV3):/$(FV3)/$(FV3) -v $(CWD)/tests:/$(FV3)/tests -v $(FV3UTIL_DIR):/external/util -v $(CWD)/external/dsl:/external/dsl -v $(CWD)/external/stencils:/external/stencils
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(PACE_IMAGE)

clean:

build:
	$(MAKE) -C .. build

cleanup_remote:
	$(MAKE) -C docker cleanup_remote

# end of image build targets which have been moved to docker/Makefile

test:
	$(MAKE) -C .. test_main

tests:
	$(MAKE) -C .. test_main

savepoint_tests:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

savepoint_tests_mpi:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

dev:
	docker run --rm -it \
		--network host \
		-v $(TEST_DATA_HOST):$(TEST_DATA_RUN_LOC) \
		-v $(CWD):/port_dev \
		$(VOLUMES) \
		$(PACE_IMAGE) bash

sync_test_data:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

sync_test_data_from_ftp:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

get_test_data:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

lint:
	pre-commit run --all-files

.PHONY: dev get_test_data tests
