WRAPPER_IMAGE ?= gcr.io/vcm-ml/fv3gfs-wrapper:v0.5.0-notebook_example
NOTEBOOK_DIR ?= $(shell pwd)/notebooks

JUPYTER_IMAGE = pace_jupyter
DOCKERFILE = Dockerfile
BUILD_ARGS = --build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE)

all: build run

run:
	docker run -p 8888:8888 -v $(NOTEBOOK_DIR) $(JUPYTER IMAGE)

build:
	docker build -t tracer_advection .
#	docker build -f $(DOCKERFILE) -t $(JUPYTER IMAGE) $(BUILD_ARGS) . --target notebook
