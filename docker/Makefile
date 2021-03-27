
# setup (use XXX=<value> make <target> to override)
include Makefile.image_names

DOCKER_BUILDKIT=1
GT4PY_DIR=../gt4py
BUILD_ARGS += \
	--network host \
	--build-arg FMS_IMAGE=$(FMS_IMAGE) \
	--build-arg ESMF_IMAGE=$(ESMF_IMAGE) \
	--build-arg MPI_IMAGE=$(MPI_IMAGE) \
	--build-arg GT4PY_DIR=${GT4PY_DIR} \
	--build-arg SERIALBOX_IMAGE=$(SERIALBOX_IMAGE) \
	--build-arg ENVIRONMENT_IMAGE=$(ENVIRONMENT_IMAGE) \
	--build-arg FORTRAN_IMAGE=$(FORTRAN_IMAGE) \
	--build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE) \
	--build-arg GT4PY_OPTIONALS=$(GT4PY_OPTIONALS) \
	--build-arg ENV_CUDA_PATH=$(ENV_CUDA_PATH)


CORE_DEPS_IMAGES = $(SERIALBOX_IMAGE) $(ENVIRONMENT_IMAGE)
DEPS_IMAGES ?= $(CORE_DEPS_IMAGES) $(FMS_IMAGE) $(ESMF_IMAGE) $(MPI_IMAGE)
GT4PY_VERSION=`cat ../GT4PY_VERSION.txt`

build: fv3core_image

dev_wrapper:
	if [ -z $(shell docker images -q $(FV3CORE_WRAPPER_IMAGE)) ]; then \
		$(MAKE) fv3core_wrapper_image; \
	fi
	FV3CORE_WRAPPER_IMAGE=$(FV3CORE_WRAPPER_IMAGE) docker-compose run fv3core_wrapper bash

.PHONY: dev fortran_image wrapper_image fv3core_image build build_deps push_deps pull_deps get_gt4py

get_gt4py:
	if [ ! -f ${GT4PY_DIR}/pyproject.toml ]  || \
	[ "$(shell cd ${GT4PY_DIR} && git describe --tags )" != "${GT4PY_VERSION}"  ]; then \
                rm -rf ${GT4PY_DIR} && \
	  	git clone https://github.com/VulcanClimateModeling/gt4py.git ${GT4PY_DIR} && \
	  	cd ${GT4PY_DIR} && git checkout ${GT4PY_VERSION};\
	fi

fortran_image:
	docker build \
		$(BUILD_ARGS) \
		-f $(DEPENDENCIES_DOCKERFILE) \
		-t $(FORTRAN_IMAGE) \
		../external/fv3gfs-fortran

# builds only the fortran stages with no external images
# useful for debugging dockerfile targets in fortran model compilation
full_fortran_image:
	FMS_IMAGE=fv3gfs-fms ESMF_IMAGE=fv3gfs-esmf MPI_IMAGE=fv3gfs-mpi $(MAKE) fortran_image

 ## build production container image
fv3core_image: get_gt4py
	DOCKER_BUILDKIT=1 docker build \
		$(BUILD_ARGS) \
		-f $(FV3CORE_DOCKERFILE) \
		-t $(FV3CORE_IMAGE) \
		--target=$(FV3CORE_TARGET) \
		..

fv3core_wrapper_image: fortran_image
	FV3CORE_TARGET=fv3core_wrapper FV3CORE_IMAGE=$(FV3CORE_WRAPPER_IMAGE) $(MAKE) fv3core_image

build_deps: build_core_deps ## build container images of all dependencies
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(FMS_IMAGE) $(BUILD_ARGS) --target fv3gfs-fms ../external/fv3gfs-fortran
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(ESMF_IMAGE) $(BUILD_ARGS) --target fv3gfs-esmf ../external/fv3gfs-fortran

build_core_deps: ## build container images of dependencies
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(MPI_IMAGE) $(BUILD_ARGS) --target fv3gfs-mpi ../external/fv3gfs-fortran
	$(MAKE) build_core_env
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(SERIALBOX_IMAGE) $(BUILD_ARGS) --target fv3gfs-environment-serialbox .

build_core_env:
	 docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(ENVIRONMENT_IMAGE) $(BUILD_ARGS) --target fv3core-environment .

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

pull_core_deps:
	DEPS_IMAGES="$(CORE_DEPS_IMAGES)" make pull_deps

pull_deps_if_needed: ## pull images of any are missing
	for IMAGE in $(DEPS_IMAGES); \
	do \
		if [ -z $$(docker images -q $$IMAGE) ]; then \
			docker pull $$IMAGE; \
		fi; \
	done

pull_core_deps_if_needed:
	DEPS_IMAGES="$(CORE_DEPS_IMAGES)" make pull_deps_if_needed

clean_deps: ## pull container images of dependencies from GCP (for faster builds)
	docker rmi $(DEPS_IMAGES)

push:
	docker push $(FV3CORE_IMAGE)

pull:
	docker pull $(FV3CORE_IMAGE)

cleanup_remote:
	gcloud container images delete -q --force-delete-tags $(FV3CORE_IMAGE)
