WRAPPER_IMAGE ?= 
NOTEBOOK_DIR ?= $(shell pwd)/tracer_advection

JUPYTER_IMAGE = pace_jupyter
DOCKERFILE = Dockerfile
BUILD_ARGS = --build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE)

all: build run

run:
	docker run -p 8888:8888 -v $(NOTEBOOK_DIR) $(JUPYTER IMAGE)

build:
	docker build -f $(DOCKERFILE)
#	docker build -f $(DOCKERFILE) -t $(JUPYTER IMAGE) $(BUILD_ARGS) . --target notebook