include docker/Makefile.image_names

GCR_URL = us.gcr.io/vcm-ml
REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_standard
FORTRAN_SERIALIZED_DATA_VERSION=7.2.6
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
FV3UTIL_DIR=$(CWD)/external/fv3gfs-util

FV3=fv3core
FV3_PATH ?=/$(FV3)

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
	VOLUMES += -v $(CWD)/$(FV3):/$(FV3)/$(FV3) -v $(CWD)/tests:/$(FV3)/tests -v $(FV3UTIL_DIR):/usr/src/fv3gfs-util
endif
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(CUDA_FLAGS) $(FV3CORE_IMAGE)

clean:
	find . -name ""
	$(RM) -rf examples/wrapped/output/*
	$(MAKE) -C external/fv3gfs-wrapper clean
	$(MAKE) -C external/fv3gfs-fortran clean

update_submodules_base:
	if [ ! -f $(FV3UTIL_DIR)/requirements.txt  ]; then \
		git submodule update --init external/fv3gfs-util ; \
	fi

update_submodules_venv: update_submodules_base
	if [ ! -f $(CWD)/external/daint_venv/install.sh  ]; then \
                git submodule update --init external/daint_venv; \
        fi

constraints.txt: requirements.txt requirements/requirements_wrapper.txt requirements/requirements_lint.txt
	pip-compile $^ --output-file constraints.txt
	sed -i '' '/^git+https/d' constraints.txt
# Image build instructions have moved to docker/Makefile but are kept here for backwards-compatibility

build_environment: update_submodules_base
	$(MAKE) -C docker build_core_deps

build_cuda_environment: build_environment
	CUDA=y $(MAKE) -C docker build_core_env

build_wrapped_environment:
	$(MAKE) -C external/fv3gfs-wrapper build-docker
	docker build \
		--network host \
		-f $(CWD)/docker/Dockerfile.build_environment \
		-t $(WRAPPER_INSTALL_IMAGE) \
		--target $(FV3_INSTALL_TARGET) \
		--build-arg BASE_IMAGE=$(WRAPPER_IMAGE) \
		.

build: update_submodules_base
	if [ $(PULL) == True ]; then \
		$(MAKE) pull_environment_if_needed; \
	else \
		$(MAKE) build_environment; \
	fi
	$(MAKE) -C docker fv3core_image

build_wrapped: update_submodules_base
	if [ $(PULL) == True ]; then \
		$(MAKE) pull_wrapped_environment_if_needed; \
	else \
		$(MAKE) build_wrapped_environment; \
	fi
	$(MAKE) -C docker fv3core_wrapper_image

pull_environment_if_needed:
	$(MAKE) -C docker pull_core_deps_if_needed


pull_wrapped_environment_if_needed:
	if [ -z $(shell docker images -q $(WRAPPER_INSTALL_IMAGE)) ]; then \
		docker pull $(WRAPPER_INSTALL_IMAGE); \
	fi

pull_environment:
	$(MAKE) -C docker pull_core_deps

push_environment:
	$(MAKE) -C docker push_deps

rebuild_environment: build_environment push_environment

push_core:
	$(MAKE) -C docker push

pull_core:
	$(MAKE) -C docker pull

tar_core:
	$(MAKE) -C docker tar_core

sarus_load_tar:
	$(MAKE) -C docker sarus_load_tar

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
		$(FV3CORE_IMAGE) bash

dev_wrapper:
	$(MAKE) -C docker dev_wrapper

test_base:
ifneq ($(findstring docker,$(CONTAINER_CMD)),)
    ifeq ($(DEV),n)
	$(MAKE) build
    endif
endif
	$(CONTAINER_CMD) bash -c "pip list && $(PYTEST_CMD)"

sync_test_data:
	mkdir -p $(TEST_DATA_HOST) && gsutil -m rsync -r $(DATA_BUCKET) $(TEST_DATA_HOST)

push_python_regressions:
	gsutil -m cp -r $(TEST_DATA_HOST)/python_regressions $(DATA_BUCKET)python_regressions

get_test_data:
	if [ ! -f "$(TEST_DATA_HOST)/input.nml" ] || \
	[ "$$(gsutil cat $(DATA_BUCKET)md5sums.txt)" != "$$(cat $(TEST_DATA_HOST)/md5sums.txt)"  ]; then \
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

gt4py_tests_gpu:
	CUDA=y make build && \
        docker run --gpus all $(FV3CORE_IMAGE) python3 -m pytest -k "(not gtx86) and (not gtmc) and (not gtcuda)" -x gt4py/tests

.PHONY: update_submodules_base update_submodules_venv build_environment build dev dev_tests dev_tests_mpi flake8 lint get_test_data unpack_test_data \
	 list_test_data_options pull_environment pull_test_data push_environment \
	rebuild_environment reformat run_tests_sequential run_tests_parallel test_base test_base_parallel \
	tests push_core pull_core tar_core sarus_load_tar cleanup_remote
