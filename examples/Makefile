APP_NAME ?=pace_examples
PORT ?=8888
BUILD_OPTIONS ?=
RUN_OPTIONS ?=
CWD =$(shell pwd)

.PHONY: help build run up stop

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

build: ## Build the image
	docker build $(BUILD_OPTIONS) -t $(APP_NAME) -f Dockerfile ..

build-nc: ## Build the image without caching
	$(MAKE) BUILD_OPTIONS="--no-cache" build

run: ## Run container
	docker run -i -t --rm $(RUN_OPTIONS) -p=$(PORT):$(PORT) --name="$(APP_NAME)" $(APP_NAME)

dev: ## Run container and mount local directory for development
	$(MAKE) RUN_OPTIONS="-v $(CWD)/notebooks:/notebooks" run

up: build run ## Build image and run container

stop: ## Stop and remove running container
	docker stop $(APP_NAME)
	docker rm $(APP_NAME)
