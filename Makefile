GCR_URL = us.gcr.io/vcm-ml
REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_standard
FV3CORE_VERSION=0.1.0
FORTRAN_SERIALIZED_DATA_VERSION=7.1.1

SHELL=/bin/bash
CWD=$(shell pwd)
TEST_ARGS ?=-v -s -rsx
PULL ?=True
NUM_RANKS ?=6
VOLUMES ?=
MOUNTS ?=
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?="--rm"
TEST_DATA_HOST ?=$(CWD)/test_data/$(EXPERIMENT)
FV3=fv3core
FV3UTIL_DIR=$(CWD)/external/fv3util
FV3UTIL_DIR=$(CWD)/external/fv3gfs-util
FV3_INSTALL_TAG ?= develop
FV3_INSTALL_TARGET=$(FV3)-install
FV3_INSTALL_IMAGE=$(GCR_URL)/$(FV3_INSTALL_TARGET):$(FV3_INSTALL_TAG)
FV3_TAG ?= $(FV3CORE_VERSION)-$(FV3_INSTALL_TAG)
FV3_IMAGE ?=$(GCR_URL)/$(FV3):$(FV3_TAG)

TEST_DATA_CONTAINER=/test_data
PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')
TEST_DATA_TARFILE=dat_files.tar.gz
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)
CORE_TAR=$(FV3_TAG).tar
CORE_BUCKET_LOC=gs://vcm-jenkins/$(CORE_TAR)
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS)
BASE_INSTALL?=$(FV3)-install-serialbox
DEV_MOUNTS = '-v $(CWD)/$(FV3):/$(FV3)/$(FV3) -v $(CWD)/tests:/$(FV3)/tests -v $(FV3UTIL_DIR):/usr/src/fv3gfs-util -v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER)'

clean:
	find . -name ""

update_submodules:
	if [ ! -f $(FV3UTIL_DIR)/requirements.txt  ]; then \
		git submodule update --init --recursive; \
	fi


build_environment:
	DOCKER_BUILDKIT=1 docker build \
		--network host \
		--build-arg MIDBASE=$(BASE_INSTALL) \
		-f $(CWD)/docker/Dockerfile.build_environment \
		-t $(FV3_INSTALL_IMAGE) \
		--target $(FV3_INSTALL_TARGET) \
		.

build: update_submodules
	if [ $(PULL) == True ]; then \
		$(MAKE) pull_environment_if_needed; \
	else \
		$(MAKE) build_environment; \
	fi
	docker build \
		--network host \
		--build-arg build_image=$(FV3_INSTALL_IMAGE) \
		-f $(CWD)/docker/Dockerfile \
		-t $(FV3_IMAGE) \
		.

pull_environment_if_needed:
	if [ -z $(shell docker images -q $(FV3_INSTALL_IMAGE)) ]; then \
		docker pull $(FV3_INSTALL_IMAGE); \
	fi

pull_environment:
	docker pull $(FV3_INSTALL_IMAGE)

push_environment:
	docker push $(FV3_INSTALL_IMAGE)

rebuild_environment: build_environment
	$(MAKE) push_environment

push_core:
	docker push $(FV3_IMAGE)

pull_core:
	docker pull $(FV3_IMAGE)

tar_core:
	docker save $(FV3_IMAGE) -o $(CORE_TAR)
	gsutil copy $(CORE_TAR) $(CORE_BUCKET_LOC)

sarus_load_tar:
	export FOUND_IMAGE=`sarus images | grep $(FV3_TAG)`
	if [ -z $(FOUND_IMAGE) ] && [ ! -f `pwd`/$(CORE_TAR) ]; then \
		gsutil copy $(CORE_BUCKET_LOC) . && \
		sarus load ./$(CORE_TAR) $(FV3_TAG); \
        fi

cleanup_remote:
	gsutil rm $(CORE_BUCKET_LOC)
	gcloud container images delete -q --force-delete-tags $(FV3_IMAGE)

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
		$(FV3_IMAGE)


dev_tests:
	VOLUMES=$(DEV_MOUNTS) $(MAKE) test_base

dev_tests_mpi:
	VOLUMES=$(DEV_MOUNTS) $(MAKE) test_base_parallel

dev_test_mpi: dev_tests_mpi


dev_tests_mpi_host:
	MOUNTS=$(DEV_MOUNTS) $(MAKE) run_tests_parallel_host

test_base:
	$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(MOUNTS) \
	$(FV3_IMAGE) bash -c "pip list && pytest --data_path=$(TEST_DATA_CONTAINER) $(TEST_ARGS) /$(FV3)/tests"

test_base_parallel:
	$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(MOUNTS) $(FV3_IMAGE) \
	$(MPIRUN_CALL) \
	bash -c "pip list && pytest --data_path=$(TEST_DATA_CONTAINER) $(TEST_ARGS) -m parallel /$(FV3)/tests"


run_tests_sequential:
	VOLUMES='--mount=type=bind,source=$(TEST_DATA_HOST),destination=$(TEST_DATA_CONTAINER) --mount=type=bind,source=$(CWD)/.jenkins,destination=/.jenkins' \
	$(MAKE) test_base

run_tests_parallel:
	VOLUMES='--mount=type=bind,source=$(TEST_DATA_HOST),destination=$(TEST_DATA_CONTAINER) --mount=type=bind,source=$(CWD)/.jenkins,destination=/.jenkins' \
	$(MAKE) test_base_parallel

sync_test_data:
	mkdir -p $(TEST_DATA_HOST) && gsutil -m rsync $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/ $(TEST_DATA_HOST)

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
	pre-commit run
	# pre-commit runs black for now. Will also run flake8 eventually.
	# black --diff --check $(PYTHON_FILES) $(PYTHON_INIT_FILES)
	# disable flake8 tests for now, re-enable when dycore is "running"
	#@flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	#@flake8 --ignore=F401 $(PYTHON_INIT_FILES)
	# @echo "LINTING SUCCESSFUL"

.PHONY: update_submodules build_environment build dev dev_tests dev_tests_mpi flake8 lint get_test_data unpack_test_data \
	 list_test_data_options pull_environment pull_test_data push_environment \
	rebuild_environment reformat run_tests_sequential run_tests_parallel test_base test_base_parallel \
	tests update_submodules push_core pull_core tar_core sarus_load_tar cleanup_remote
