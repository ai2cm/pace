include docker/Makefile.image_names

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm


build:
	PULL=PULL $(MAKE) -C docker fv3gfs_image

dev:
	docker run --rm -it \
		--network host \
		-v $(CWD):/port_dev \
		$(FV3GFS_IMAGE) bash

test_util:
	$(MAKE) -C fv3gfs-util test
	
savepoint_tests:
	$(MAKE) -C fv3core savepoint_tests

savepoint_tests_mpi:
	$(MAKE) -C fv3core savepoint_tests_mpi

dependencies.svg:
	dot -Tsvg dependencies.dot -o $@

constraints.txt: fv3core/requirements.txt fv3core/requirements/requirements_wrapper.txt fv3core/requirements/requirements_lint.txt fv3gfs-util/requirements.txt fv3gfs-physics/requirements.txt
	pip-compile $^ --output-file constraints.txt
