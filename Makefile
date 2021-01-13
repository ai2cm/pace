include docker/Makefile.image_names

GCR_URL = us.gcr.io/vcm-ml
REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_standard
FORTRAN_SERIALIZED_DATA_VERSION=7.2.3
WRAPPER_IMAGE = us.gcr.io/vcm-ml/fv3gfs-wrapper:gnu9-mpich314-nocuda
DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
NUM_RANKS ?=6
VOLUMES ?=
MOUNTS ?=
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
TEST_DATA_HOST ?=$(CWD)/test_data/$(EXPERIMENT)
FV3UTIL_DIR=$(CWD)/external/fv3gfs-util

FV3=fv3core
ifeq ($(CONTAINER_ENGINE),sarus)
	FV3_IMAGE = load/library/$(SARUS_FV3CORE_IMAGE)
else ifeq ($(CONTAINER_ENGINE),srun sarus)
	FV3_IMAGE = load/library/$(SARUS_FV3CORE_IMAGE)
else
	FV3_IMAGE ?= $(FV3CORE_IMAGE)
endif

TEST_DATA_CONTAINER=/test_data
PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')
TEST_DATA_TARFILE=dat_files.tar.gz
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)
CORE_TAR=$(SARUS_FV3CORE_IMAGE).tar
CORE_BUCKET_LOC=gs://vcm-jenkins/$(CORE_TAR)
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS)
BASE_INSTALL?=$(FV3)-install-serialbox
DEV_MOUNTS = '-v $(CWD)/$(FV3):/$(FV3)/$(FV3) -v $(CWD)/tests:/$(FV3)/tests -v $(FV3UTIL_DIR):/usr/src/fv3gfs-util -v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER)'

clean:
	find . -name ""
	$(RM) -rf examples/wrapped/output/*
	$(MAKE) -C external/fv3gfs-wrapper clean
	$(MAKE) -C external/fv3gfs-fortran clean

update_submodules:
	if [ ! -f $(FV3UTIL_DIR)/requirements.txt  ]; then \
		git submodule update --init --recursive; \
	fi

constraints.txt: requirements.txt requirements_wrapper.txt requirements_lint.txt
	pip-compile $^ --output-file constraints.txt
	sed -i '' '/^git+https/d' constraints.txt
# Image build instructions have moved to docker/Makefile but are kept here for backwards-compatibility

build_environment: update_submodules
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

build: update_submodules
	if [ $(PULL) == True ]; then \
		$(MAKE) pull_environment_if_needed; \
	else \
		$(MAKE) build_environment; \
	fi
	$(MAKE) -C docker fv3core_image

build_wrapped: update_submodules
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

tests: build
	$(MAKE) get_test_data
	$(MAKE) run_tests_sequential

test: tests

tests_mpi: build
	$(MAKE) get_test_data
	$(MAKE) run_tests_parallel

test_mpi: tests_mpi

dev:
	docker run --rm -it \
		--network host \
		-v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER) \
		-v $(CWD):/port_dev \
		$(FV3_IMAGE) bash

dev_wrapper:
	$(MAKE) -C docker dev_wrapper

dev_tests:
	VOLUMES=$(DEV_MOUNTS) $(MAKE) test_base

dev_tests_mpi:
	VOLUMES=$(DEV_MOUNTS) $(MAKE) test_base_parallel

dev_test_mpi: dev_tests_mpi

dev_tests_mpi_host:
	MOUNTS=$(DEV_MOUNTS) $(MAKE) run_tests_parallel_host

test_base:
	$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(MOUNTS) $(CUDA_FLAGS) \
	$(FV3_IMAGE) bash -c "pip list && pytest --data_path=$(TEST_DATA_CONTAINER) $(TEST_ARGS) /$(FV3)/tests"

test_base_parallel:
	$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(MOUNTS) $(CUDA_FLAGS) $(FV3_IMAGE) \
	bash -c "pip list && $(MPIRUN_CALL) pytest --data_path=$(TEST_DATA_CONTAINER) $(TEST_ARGS) -m parallel /$(FV3)/tests"

run_tests_sequential:
	VOLUMES='--mount=type=bind,source=$(TEST_DATA_HOST),destination=$(TEST_DATA_CONTAINER) --mount=type=bind,source=$(CWD)/.jenkins,destination=/.jenkins' \
	$(MAKE) test_base

run_tests_parallel:
	VOLUMES='--mount=type=bind,source=$(TEST_DATA_HOST),destination=$(TEST_DATA_CONTAINER) --mount=type=bind,source=$(CWD)/.jenkins,destination=/.jenkins' \
	$(MAKE) test_base_parallel

sync_test_data:
	mkdir -p $(TEST_DATA_HOST) && gsutil -m rsync -r $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/ $(TEST_DATA_HOST)

push_python_regressions:
	gsutil -m cp -r $(TEST_DATA_HOST)/python_regressions $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/python_regressions

get_test_data:
	if [ ! -f "$(TEST_DATA_HOST)/input.nml" ]; then \
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
        docker run --gpus all $(FV3_IMAGE) python3 -m pytest -x gt4py/

.PHONY: update_submodules build_environment build dev dev_tests dev_tests_mpi flake8 lint get_test_data unpack_test_data \
	 list_test_data_options pull_environment pull_test_data push_environment \
	rebuild_environment reformat run_tests_sequential run_tests_parallel test_base test_base_parallel \
	tests update_submodules push_core pull_core tar_core sarus_load_tar cleanup_remote
