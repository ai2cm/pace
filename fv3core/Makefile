include ../docker/Makefile.image_names

GCR_URL = us.gcr.io/vcm-ml
REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_standard
FORTRAN_SERIALIZED_DATA_VERSION=8.1.1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=n
NUM_RANKS ?=6
VOLUMES ?=
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
RUN_FLAGS += -e FV3_DACEMODE=$(FV3_DACEMODE)
TEST_ARGS ?=
TARGET ?=dycore
TEST_DATA_ROOT ?=$(CWD)/test_data/
TEST_DATA_HOST ?=$(TEST_DATA_ROOT)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/$(TARGET)
FV3UTIL_DIR=$(CWD)/external/pace-util

FV3=fv3core
FV3_PATH ?=/pace/$(FV3)

ifeq ($(DEV),y)
	VOLUMES += -v $(CWD):$(FV3_PATH) -v $(FV3UTIL_DIR):/usr/src/pace-util -v $(CWD)/external/stencils:/external/stencils  -v $(CWD)/external/dsl:/external/dsl
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(CUDA_FLAGS) $(PACE_IMAGE)

ifneq (,$(findstring $(FV3GFS_IMAGE),$(CONTAINER_CMD)))
	TEST_DATA_RUN_LOC =/test_data
else
	TEST_DATA_RUN_LOC=$(TEST_DATA_HOST)
endif

PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')
TEST_DATA_TARFILE=dat_files.tar.gz
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS) --oversubscribe --mca btl_vader_single_copy_mechanism=none
BASE_INSTALL?=$(FV3)-install-serialbox
DATA_BUCKET= $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/$(TARGET)/

TEST_TYPE=$(word 3, $(subst _, ,$(EXPERIMENT)))
THRESH_ARGS=--threshold_overrides_file=$(FV3_PATH)/tests/savepoint/translate/overrides/$(TEST_TYPE).yaml
PYTEST_MAIN=pytest $(TEST_ARGS) $(FV3_PATH)/tests/main
PYTEST_SEQUENTIAL=pytest --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) $(FV3_PATH)/tests/savepoint
# we can't rule out a deadlock if one test fails, so we must set maxfail=1 for parallel tests
PYTEST_PARALLEL=$(MPIRUN_CALL) python -m mpi4py -m pytest --maxfail=1 --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) -m parallel $(FV3_PATH)/tests/savepoint
ifeq ($(DEV),y)
	VOLUMES += -v $(CWD)/$(FV3):/$(FV3)/$(FV3) -v $(CWD)/tests:/$(FV3)/tests -v $(FV3UTIL_DIR):/external/pace-util -v $(CWD)/external/dsl:/external/dsl -v $(CWD)/external/stencils:/external/stencils
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(CUDA_FLAGS) $(PACE_IMAGE)

clean:

update_submodules_venv:
	if [ ! -f $(CWD)/external/daint_venv/install.sh  ]; then \
                git submodule update --init external/daint_venv; \
        fi

build:
	$(MAKE) -C .. build

cleanup_remote:
	$(MAKE) -C docker cleanup_remote

# end of image build targets which have been moved to docker/Makefile

test: tests

tests:
	PYTEST_CMD="$(PYTEST_MAIN)" $(MAKE) test_base

savepoint_tests:
	$(MAKE) get_test_data
	VOLUMES='$(VOLUMES) -v $(TEST_DATA_HOST):$(TEST_DATA_RUN_LOC)' \
	PYTEST_CMD="$(PYTEST_SEQUENTIAL)" $(MAKE) test_base

savepoint_tests_mpi:
	$(MAKE) get_test_data
	VOLUMES='$(VOLUMES) -v $(TEST_DATA_HOST):$(TEST_DATA_RUN_LOC)' \
	PYTEST_CMD="$(PYTEST_PARALLEL)" $(MAKE) test_base

dev:
	docker run --rm -it \
		--network host \
		-v $(TEST_DATA_HOST):$(TEST_DATA_RUN_LOC) \
		-v $(CWD):/port_dev \
		$(VOLUMES) \
		$(PACE_IMAGE) bash

test_base:
ifneq ($(findstring docker,$(CONTAINER_CMD)),)
    ifeq ($(DEV),n)
	$(MAKE) build
    endif
endif
	$(CONTAINER_CMD) bash -c "pip list && cd $(FV3_PATH) && $(PYTEST_CMD)"

sync_test_data:
	mkdir -p $(TEST_DATA_HOST) && gsutil -m rsync -r $(DATA_BUCKET) $(TEST_DATA_HOST)

push_python_regressions:
	gsutil -m cp -r $(TEST_DATA_HOST)/python_regressions $(DATA_BUCKET)python_regressions

get_test_data:
	if [ ! -f "$(TEST_DATA_HOST)/input.nml" ] || \
	[ "$$(gsutil cp $(DATA_BUCKET)md5sums.txt -)" != "$$(cat $(TEST_DATA_HOST)/md5sums.txt)"  ]; then \
	rm -rf $(TEST_DATA_HOST) && \
	$(MAKE) sync_test_data && \
	$(MAKE) unpack_test_data ;\
	fi

unpack_test_data:
	if [ -f $(TEST_DATA_TARPATH) ]; then \
	cd $(TEST_DATA_HOST) && tar -xf $(TEST_DATA_TARFILE) && \
	rm $(TEST_DATA_TARFILE); fi

list_test_data_options:
	gsutil ls $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)

lint:
	pre-commit run --all-files

.PHONY: update_submodules_venv dev dev_tests dev_tests_mpi flake8 lint get_test_data unpack_test_data \
	 list_test_data_options pull_environment pull_test_data push_environment \
	reformat run_tests_sequential run_tests_parallel test_base test_base_parallel \
	tests cleanup_remote
