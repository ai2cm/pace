include ../docker/Makefile.image_names

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=n
NUM_RANKS ?=6
pace=pace
PACE_PATH ?=/$(pace)
RUN_FLAGS ?=--rm
CONTAINER_ENGINE ?=docker
EXPERIMENT ?=c12_6ranks_baroclinic_dycore_microphysics
DRIVER_VOLUME = -v $(CWD):/port_dev
ifeq ($(DEV),y)
	DRIVER_VOLUME = -v $(CWD):/driver
endif
VOLUMES = $(DRIVER_VOLUME) -v $(CWD)/../test_data/$(EXPERIMENT):/test_data
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS)
CONTAINER_CMD?=$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(FV3GFS_IMAGE)
PYTEST_CMD=pytest /driver/tests/
build:
	PULL=$(PULL) $(MAKE) -C ../docker fv3gfs_image

test:
ifneq ($(findstring docker,$(CONTAINER_CMD)),)
    ifeq ($(DEV),n)
	$(MAKE) build
    endif
endif
	$(CONTAINER_CMD) bash -c "$(PYTEST_CMD) -vv -s "

get_test_data:
	$(MAKE) -C ../fv3gfs-physics get_test_data

test_serialized_init: get_test_data
ifneq ($(findstring docker,$(CONTAINER_CMD)),)
    ifeq ($(DEV),n)
	$(MAKE) build
    endif
endif
	$(CONTAINER_CMD) bash -c "$(MPIRUN_CALL) python3 driver/pace/driver/run.py /test_data 3 numpy serialized_data"

test_mpi:
	$(MPIRUN_CALL) python3 -m mpi4py -m pace.driver.run examples/configs/baroclinic_c12.yaml
