include docker/Makefile.image_names

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=n
VOLUMES ?=
NUM_RANKS ?=6
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
CHECK_CHANGED_SCRIPT=$(CWD)/changed_from_main.py
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS)

REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_baroclinic_dycore_microphysics
FORTRAN_SERIALIZED_DATA_VERSION=integration-7.2.5
TEST_DATA_TARFILE=dat_files.tar.gz
TEST_DATA_HOST ?=$(CWD)/test_data/$(EXPERIMENT)
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)
FV3UTIL_DIR=$(CWD)/external/fv3gfs-util
DATA_BUCKET= $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/
TEST_DATA_RUN_LOC ?=/test_data

pace=pace
pace_PATH ?=/$(pace)
TEST_HOST_LOC=$(CWD)/tests
TEST_RUN_LOC ?=$(pace_PATH)/tests
THRESH_ARGS=--threshold_overrides_file=$(pace_PATH)/tests/savepoint/translate/overrides/baroclinic.yaml
PYTEST_SEQUENTIAL=pytest --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) $(pace_PATH)/tests/savepoint
PYTEST_PARALLEL=$(MPIRUN_CALL) python -m mpi4py -m pytest --maxfail=1 --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) -m parallel $(pace_PATH)/tests/savepoint
ifeq ($(DEV),y)
	VOLUMES += -v $(CWD)/$(pace):/$(pace)/$(pace) -v $(CWD)/tests:/$(pace)/tests -v $(FV3UTIL_DIR):/usr/src/fv3gfs-util
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(CUDA_FLAGS) $(FV3GFS_IMAGE)

build:
	PULL=PULL $(MAKE) -C docker fv3gfs_image

dev:
	docker run --rm -it \
		--network host \
		-v $(CWD):/port_dev \
		$(FV3GFS_IMAGE) bash

test_util:
	if [ $(shell $(CHECK_CHANGED_SCRIPT) fv3gfs-util) != false ]; then \
		$(MAKE) -C fv3gfs-util test; \
	fi

savepoint_tests:
	$(MAKE) -C fv3core $@

savepoint_tests_mpi:
	$(MAKE) -C fv3core $@

dependencies.svg: dependencies.dot
	dot -Tsvg $< -o $@

constraints.txt: fv3core/requirements.txt fv3core/requirements/requirements_wrapper.txt fv3core/requirements/requirements_lint.txt fv3gfs-util/requirements.txt fv3gfs-physics/requirements.txt
	pip-compile $^ --output-file constraints.txt

sync_test_data:
	mkdir -p $(TEST_DATA_HOST) && gsutil -m rsync -r $(DATA_BUCKET) $(TEST_DATA_HOST)

unpack_test_data:
	if [ -f $(TEST_DATA_TARPATH) ]; then \
	cd $(TEST_DATA_HOST) && tar -xf $(TEST_DATA_TARFILE) && \
	rm $(TEST_DATA_TARFILE); fi

get_test_data:
	if [ ! -f "$(TEST_DATA_HOST)/input.nml" ] || \
	[ "$$(gsutil cat $(DATA_BUCKET)md5sums.txt)" != "$$(cat $(TEST_DATA_HOST)/md5sums.txt)"  ]; then \
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