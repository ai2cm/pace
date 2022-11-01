IMAGE_NAME ?=gcr.io/vcm-ml/pace_notebook_examples
APP_NAME ?=pace_notebook_examples
PORT ?=8888
BUILD_OPTIONS ?=
RUN_OPTIONS ?=
CMD?=
CWD =$(shell pwd)

.PHONY: help build build-nc run dev up stop

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

build: ## Build the image
	docker build $(BUILD_OPTIONS) -t $(IMAGE_NAME) -f Dockerfile ..

build-nc: ## Build the image without caching
	$(MAKE) BUILD_OPTIONS="--no-cache" build

run: ## Run container
	docker run -i -t --rm $(RUN_OPTIONS) -p=$(PORT):$(PORT) --name="$(APP_NAME)" $(IMAGE_NAME) $(CMD)

dev: ## Run container and mount local directory for development
	RUN_OPTIONS="-v $(CWD)/notebooks:/notebooks" $(MAKE) run

enter:
	CMD="/bin/bash" $(MAKE) run

up: build run ## Build image and run container

stop: ## Stop and remove running container
	docker stop $(APP_NAME)
	docker rm $(APP_NAME)
