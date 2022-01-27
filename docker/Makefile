SHELL := /bin/bash
# setup (use XXX=<value> make <target> to override)
include Makefile.image_names

DOCKER_BUILDKIT=1
BUILD_ARGS += \
	--network host \
	--build-arg MPI_IMAGE=$(MPI_IMAGE) \
	--build-arg SERIALBOX_IMAGE=$(SERIALBOX_IMAGE) \
	--build-arg ENVIRONMENT_IMAGE=$(ENVIRONMENT_IMAGE) \
	--build-arg GT4PY_OPTIONALS=$(GT4PY_OPTIONALS) \
	--build-arg ENV_CUDA_PATH=$(ENV_CUDA_PATH)


DEPS_IMAGES = $(SERIALBOX_IMAGE) $(ENVIRONMENT_IMAGE) $(MPI_IMAGE)
PULL ?=True

build: fv3gfs_image

.PHONY: dev fv3gfs_image build build_deps push_deps pull_deps

 ## build production container image
fv3gfs_image:
	if [ $(PULL) == True ]; then \
		$(MAKE) pull_deps_if_needed; \
	else \
		$(MAKE) build_deps; \
	fi
	DOCKER_BUILDKIT=1 docker build \
		$(BUILD_ARGS) \
		-f $(FV3GFS_DOCKERFILE) \
		-t $(FV3GFS_IMAGE) \
		..

build_deps: ## build container images of dependencies
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(MPI_IMAGE) $(BUILD_ARGS) --target fv3gfs-mpi .
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(ENVIRONMENT_IMAGE) $(BUILD_ARGS) --target fv3gfs-environment .
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(SERIALBOX_IMAGE) $(BUILD_ARGS) --target fv3gfs-environment-serialbox .

push_deps: ## push container images of dependencies to GCP
	for IMAGE in $(DEPS_IMAGES); \
	do \
		if [ ! -z $$(docker images -q $$IMAGE) ]; then \
			docker push $$IMAGE; \
		fi; \
	done

pull_deps: ## pull container images of dependencies from GCP (for faster builds)
	for IMAGE in $(DEPS_IMAGES); \
	do \
		docker pull $$IMAGE; \
	done

pull_deps_if_needed: ## pull images of any are missing
	for IMAGE in $(DEPS_IMAGES); \
	do \
		if [ -z $$(docker images -q $$IMAGE) ]; then \
			docker pull $$IMAGE; \
		fi; \
	done

clean_deps: ## pull container images of dependencies from GCP (for faster builds)
	docker rmi $(DEPS_IMAGES)

push:
	docker push $(FV3GFS_IMAGE)

pull:
	docker pull $(FV3GFS_IMAGE)

cleanup_remote:
	gcloud container images delete -q --force-delete-tags $(FV3GFS_IMAGE)
