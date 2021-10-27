include docker/Makefile.image_names
EXPERIMENT ?=c12_6ranks_baroclinic_dycore_microphysics
DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
TEST_DATA_RUN_LOC=/port_dev/fv3core/test_data/$(EXPERIMENT)
TEST_TYPE=$(word 3, $(subst _, ,$(EXPERIMENT)))
TEST_ARGS ?=
FV3_PATH ?=/port_dev
THRESH_ARGS=--threshold_overrides_file=$(FV3_PATH)/tests/savepoint/translate/overrides/$(TEST_TYPE).yaml
PYTEST_MAIN=pytest $(TEST_ARGS) $(FV3_PATH)/tests/main
PYTEST_SEQUENTIAL=pytest --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) $(FV3_PATH)/tests/savepoint
PYTEST_PARALLEL=$(MPIRUN_CALL) python -m mpi4py -m pytest --maxfail=1 --data_path=$(TEST_DATA_RUN_LOC) $(TEST_ARGS) $(THRESH_ARGS) -m parallel $(FV3_PATH)/tests/savepoint
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

constraints.txt: fv3core/requirements.txt fv3core/requirements/requirements_wrapper.txt fv3core/requirements/requirements_lint.txt fv3gfs-util/requirements.txt fv3gfs-physics/requirements.txt
	pip-compile $^ --output-file constraints.txt
savepoint_tests_top:
	docker run --rm \
		--network host \
		-v $(CWD):/port_dev \
		$(FV3GFS_IMAGE) $(PYTEST_SEQUENTIAL)

example:
	docker run --rm \
		--network host \
		-v $(CWD):/port_dev \
		$(FV3GFS_IMAGE) bash -c 'cd /port_dev &&  mpirun -np 6 python3 /port_dev/examples/dyn_phy.py'

run:
	docker run --rm \
		--network host \
		-v $(CWD):/port_dev \
		$(FV3GFS_IMAGE) bash -c 'cd /port_dev &&  mpirun -np 6 python3 /port_dev/driver/run_model.py'
