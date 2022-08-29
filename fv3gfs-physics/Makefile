include ../docker/Makefile.image_names

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=y
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
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
TEST_DATA_ROOT ?=$(ROOT_DIR)/test_data/

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
	$(MAKE) -C .. $@

sync_test_data:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

sync_test_data_from_ftp:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

get_test_data:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

physics_savepoint_tests:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

physics_savepoint_tests_mpi:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@

driver_savepoint_tests_mpi:
	TEST_DATA_ROOT=$(TEST_DATA_ROOT) TARGET=$(TARGET) EXPERIMENT=$(EXPERIMENT) $(MAKE) -C .. $@
