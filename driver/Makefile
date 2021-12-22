include ../docker/Makefile.image_names

DOCKER_BUILDKIT=1
SHELL=/bin/bash
CWD=$(shell pwd)
PULL ?=True
DEV ?=n
pace=pace
PACE_PATH ?=/$(pace)
RUN_FLAGS ?=--rm
CONTAINER_ENGINE ?=docker
VOLUMES = -v $(CWD):/port_dev
ifeq ($(DEV),y)
	VOLUMES = -v $(CWD):/driver
endif
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
