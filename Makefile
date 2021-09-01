include docker/Makefile.image_names

EXPERIMENT ?=c12_6ranks_standard
DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
CONTAINER_ENGINE ?=docker
TEST_DATA_HOST ?=$(CWD)/test_data/$(EXPERIMENT)
FV3UTIL_DIR=$(CWD)/fv3gfs-util

FV3=fv3core
FV3_PATH ?=/$(FV3)
PHY=fv3gfs-physics
PHY_PATH ?=/$(PHY)

TEST_DATA_RUN_LOC ?=/test_data

build:
	if [ $(PULL) == True ]; then \
		$(MAKE) pull_environment_if_needed; \
	else \
		$(MAKE) build_environment; \
	fi
	$(MAKE) -C docker fv3gfs_image

build_environment:
	$(MAKE) -C docker build_core_deps

pull_environment_if_needed:
	$(MAKE) -C docker pull_core_deps_if_needed

dev:
	docker run --rm -it \
		--network host \
		-v $(TEST_DATA_HOST):$(TEST_DATA_RUN_LOC) \
		-v $(CWD):/port_dev \
		$(FV3GFS_IMAGE) bash

test_util:
	$(MAKE) -C fv3gfs-util test
	
savepoint_tests:
	$(MAKE) -C fv3core savepoint_tests

savepoint_tests_mpi:
	$(MAKE) -C fv3core savepoint_tests_mpi

constraints.txt: fv3core/requirements.txt fv3core/requirements/requirements_wrapper.txt fv3core/requirements/requirements_lint.txt fv3gfs-util/requirements.txt fv3gfs-physics/requirements.txt
	pip-compile $^ --output-file constraints.txt

link_fv3core_test_data:
	rm -rf $(TEST_DATA_HOST)
	ln -s fv3core/test_data .