GCR_URL = us.gcr.io/vcm-ml
REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_standard
FV3CORE_VERSION=0.0.0
FORTRAN_SERIALIZED_DATA_VERSION=7.1.1
SHELL=/bin/bash
CWD=$(shell pwd)
TEST_ARGS ?=-v -s -rsx
PULL ?=True
NUM_RANKS ?=6
VOLUMES ?=
MOUNTS ?=

TEST_DATA_HOST ?=$(CWD)/test_data/$(EXPERIMENT)
FV3_IMAGE ?=$(GCR_URL)/fv3core:$(FV3CORE_VERSION)
FV3UTIL_DIR=$(CWD)/external/fv3util
DEV_MOUNTS = '-v $(CWD)/fv3core:/fv3core/fv3core -v $(CWD)/tests:/fv3core/tests -v $(FV3UTIL_DIR):/usr/src/fv3util'
FV3_INSTALL_TAG=master
FV3_INSTALL_TARGET=fv3core-install
FV3_INSTALL_IMAGE=$(GCR_URL)/$(FV3_INSTALL_TARGET):$(FV3_INSTALL_TAG)


TEST_DATA_CONTAINER=/test_data

PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')
TEST_DATA_TARFILE=dat_files.tar.gz
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)

clean:
	find . -name ""

update_submodules:
	if [ ! -f $(FV3UTIL_DIR)/requirements.txt  ]; then \
		git submodule update --init --recursive; \
	fi


build_environment: 
	DOCKER_BUILDKIT=1 docker build \
		--network host \
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
	MOUNTS=$(DEV_MOUNTS) $(MAKE) run_tests_sequential

dev_tests_mpi:
	MOUNTS=$(DEV_MOUNTS) $(MAKE) run_tests_parallel

dev_test_mpi: dev_tests_mpi


dev_tests_mpi_host:
	MOUNTS=$(DEV_MOUNTS) $(MAKE) run_tests_parallel_host

test_base:
	docker run --rm $(VOLUMES) $(MOUNTS) \
	$(FV3_IMAGE) pytest --data_path=$(TEST_DATA_CONTAINER) ${TEST_ARGS} /fv3core/tests

test_base_parallel:
	docker run --rm $(VOLUMES) $(MOUNTS) $(FV3_IMAGE) \
	mpirun -np $(NUM_RANKS) \
	pytest --data_path=$(TEST_DATA_CONTAINER) ${TEST_ARGS} -m parallel /fv3core/tests


run_tests_sequential:
	VOLUMES='-v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER)' \
	$(MAKE) test_base

run_tests_parallel:
	VOLUMES='-v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER)' \
	$(MAKE) test_base_parallel

get_test_data:
	if [ ! -d $(TEST_DATA_HOST) ]; then \
	mkdir -p $(TEST_DATA_HOST) && \
	gsutil -m rsync $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/ $(TEST_DATA_HOST) && \
	$(MAKE) unpack_test_data ;\
	fi

unpack_test_data:
	if [ -f $(TEST_DATA_TARPATH) ]; then \
	cd $(TEST_DATA_HOST) && tar -xf $(TEST_DATA_TARFILE) && \
	rm $(TEST_DATA_TARFILE); fi

list_test_data_options:
	gsutil ls $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)

lint:
	black --diff --check $(PYTHON_FILES) $(PYTHON_INIT_FILES)
	# disable flake8 tests for now, re-enable when dycore is "running"
	#flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	#flake8 --ignore=F401 $(PYTHON_INIT_FILES)
	@echo "LINTING SUCCESSFUL"

flake8:
	flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	flake8 --ignore=F401 $(PYTHON_INIT_FILES)

reformat:
	black $(PYTHON_FILES) $(PYTHON_INIT_FILES)

.PHONY: update_submodules build_environment build dev dev_tests dev_tests_mpi flake8 lint get_test_data unpack_test_data \
	 list_test_data_options pull_environment pull_test_data push_environment \
	rebuild_environment reformat run_tests_sequential run_tests_parallel test_base test_base_parallel \
	tests update_submodules 
