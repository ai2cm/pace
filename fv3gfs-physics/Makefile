include ../docker/Makefile.image_names

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=n
VOLUMES ?=
NUM_RANKS ?=6
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
TEST_ARGS ?=
MPIRUN_ARGS ?=--oversubscribe
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS) $(MPIRUN_ARGS)

EXPERIMENT ?=c12_6ranks_baroclinic_dycore_microphysics
TARGET ?= physics
TEST_DATA_FTP ?=in/put/abc/cosmo/fuo/pace/fv3gfs-physics
FV3UTIL_DIR=$(CWD)/../pace-util

ifeq ($(DEV),y)
	VOLUMES += -v $(CWD)/..:/pace
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(CUDA_FLAGS) $(PACE_IMAGE)

ifneq (,$(findstring $(PACE_IMAGE),$(CONTAINER_CMD)))
	TEST_DATA_RUN_LOC =/test_data
else
	TEST_DATA_RUN_LOC=$(TEST_DATA_HOST)
endif
pace=pace
PACE_PATH ?=/$(pace)
TEST_HOST_LOC=$(CWD)/tests
TEST_RUN_LOC ?=$(PACE_PATH)/fv3gfs-physics/tests
THRESH_ARGS=--threshold_overrides_file=$(PACE_PATH)/fv3gfs-physics/tests/savepoint/translate/overrides/baroclinic.yaml
PYTEST_SEQUENTIAL=pytest --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) $(PACE_PATH)/fv3gfs-physics/tests/savepoint
PYTEST_PARALLEL=$(MPIRUN_CALL) python -m mpi4py -m pytest --maxfail=1 --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) -m parallel $(PACE_PATH)/fv3gfs-physics/tests/savepoint

build:
	$(MAKE) -C .. build

sync_test_data:
	TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. sync_test_data

sync_test_data_from_ftp:
	TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. sync_test_data_from_ftp

get_test_data:
	TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. get_test_data

test_base:
ifneq ($(findstring docker,$(CONTAINER_CMD)),)
    ifeq ($(DEV),n)
	$(MAKE) build
    endif
endif
	$(CONTAINER_CMD) bash -c "pip list && $(PYTEST_CMD)"

physics_savepoint_tests:
	$(MAKE) get_test_data
	VOLUMES='$(VOLUMES) -v $(TEST_DATA_HOST):$(TEST_DATA_RUN_LOC) -v $(TEST_HOST_LOC):$(TEST_RUN_LOC)' \
	PYTEST_CMD="$(PYTEST_SEQUENTIAL)" $(MAKE) test_base

physics_savepoint_tests_mpi:
	$(MAKE) get_test_data
	VOLUMES='$(VOLUMES) -v $(TEST_DATA_HOST):$(TEST_DATA_RUN_LOC) -v $(TEST_HOST_LOC):$(TEST_RUN_LOC)' \
	PYTEST_CMD="$(PYTEST_PARALLEL)" $(MAKE) test_base

driver_savepoint_tests_mpi:
	TARGET=driver \
	$(MAKE) get_test_data
	TARGET=driver VOLUMES='  -v $(TEST_DATA_HOST)/../driver/:$(TEST_DATA_RUN_LOC) -v $(TEST_HOST_LOC):$(TEST_RUN_LOC)'  \
	PYTEST_CMD="$(PYTEST_PARALLEL)" $(MAKE) test_base
