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
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS)

REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_baroclinic_dycore_microphysics
FORTRAN_SERIALIZED_DATA_VERSION=8.0.5
TEST_DATA_TARFILE=dat_files.tar.gz
TARGET ?= physics
TEST_DATA_ROOT ?=$(CWD)/../test_data/
TEST_DATA_HOST ?=$(TEST_DATA_ROOT)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/$(TARGET)/
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)
FV3UTIL_DIR=$(CWD)/../pace-util
DATA_BUCKET ?= $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/$(TARGET)/
ifeq ($(DEV),y)
	VOLUMES += -v $(CWD)/../fv3core:/fv3core -v $(CWD)/../fv3gfs-physics:/fv3gfs-physics -v $(CWD)/../stencils:/stencils -v $(CWD)/tests:/$(pace)/tests -v $(FV3UTIL_DIR):/pace-util  -v $(CWD)/../driver:/driver -v $(CWD)/../dsl:/dsl
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(CUDA_FLAGS) $(FV3GFS_IMAGE)

ifneq (,$(findstring $(FV3GFS_IMAGE),$(CONTAINER_CMD)))
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
	PULL=$(PULL) $(MAKE) -C ../docker fv3gfs_image

sync_test_data:
	mkdir -p $(TEST_DATA_HOST) && gsutil -m rsync -r $(DATA_BUCKET) $(TEST_DATA_HOST)

unpack_test_data:
	if [ -f $(TEST_DATA_TARPATH) ]; then \
	cd $(TEST_DATA_HOST) && tar -xf $(TEST_DATA_TARFILE) && \
	rm $(TEST_DATA_TARFILE); fi

get_test_data:
	if [ ! -f "$(TEST_DATA_HOST)/input.nml" ] || \
	[ "$$(gsutil cp $(DATA_BUCKET)md5sums.txt -)" != "$$(cat $(TEST_DATA_HOST)/md5sums.txt)"  ]; then \
	rm -rf $(TEST_DATA_HOST) && \
	$(MAKE) sync_test_data && \
	$(MAKE) unpack_test_data ;\
	fi

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
