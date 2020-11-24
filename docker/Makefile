# setup (use XXX=<value> make <target> to override)
include Makefile.image_names

DOCKER_BUILDKIT=1

BUILD_ARGS += \
	--build-arg FMS_IMAGE=$(FMS_IMAGE) \
	--build-arg ESMF_IMAGE=$(ESMF_IMAGE) \
	--build-arg MPI_IMAGE=$(MPI_IMAGE) \
	--build-arg GT4PY_IMAGE=$(GT4PY_IMAGE) \
	--build-arg SERIALBOX_IMAGE=$(SERIALBOX_IMAGE) \
	--build-arg ENVIRONMENT_IMAGE=$(ENVIRONMENT_IMAGE) \
	--build-arg FORTRAN_IMAGE=$(FORTRAN_IMAGE) \
	--build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE) \
	--build-arg GT4PY_OPTIONALS=$(GT4PY_OPTIONALS) \
	--build-arg ENV_CUDA_PATH=$(ENV_CUDA_PATH)

# sarus paths
CORE_TAR=$(TAG_NAME)$(JENKINS_TAG).tar
CORE_BUCKET_LOC=gs://vcm-jenkins/$(CORE_TAR)

CORE_DEPS_IMAGES = $(MPI_IMAGE) $(SERIALBOX_IMAGE) $(GT4PY_IMAGE) $(ENVIRONMENT_IMAGE)
DEPS_IMAGES ?= $(CORE_DEPS_IMAGES) $(FMS_IMAGE) $(ESMF_IMAGE)

build: fv3core_image

dev_wrapper:
	if [ -z $(shell docker images -q $(FV3CORE_WRAPPER_IMAGE)) ]; then \
		$(MAKE) fv3core_wrapper_image; \
	fi
	FV3CORE_WRAPPER_IMAGE=$(FV3CORE_WRAPPER_IMAGE) docker-compose run fv3core_wrapper bash

.PHONY: dev fortran_image wrapper_image fv3core_image build build_deps push_deps pull_deps

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

fv3core_image: ## build production container image
	docker build \
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
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(ENVIRONMENT_IMAGE) $(BUILD_ARGS) --target fv3core-environment .
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(MPI_IMAGE) $(BUILD_ARGS) --target fv3gfs-mpi ../external/fv3gfs-fortran
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(SERIALBOX_IMAGE) $(BUILD_ARGS) --target fv3gfs-environment-serialbox .
	docker build -f $(DEPENDENCIES_DOCKERFILE) -t $(GT4PY_IMAGE) $(BUILD_ARGS) --target gt4py-src .

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

tar_core:
	docker save $(FV3CORE_IMAGE) -o $(CORE_TAR)
	gsutil copy $(CORE_TAR) $(CORE_BUCKET_LOC)
	$(RM) $(CORE_TAR)

sarus_load_tar:
	export FOUND_IMAGE=`sarus images | grep $(TAG_NAME)`
	if [ -z $(FOUND_IMAGE) ] && [ ! -f `pwd`/$(CORE_TAR) ]; then \
		gsutil copy $(CORE_BUCKET_LOC) . && \
		sarus load ./$(CORE_TAR) $(SARUS_FV3CORE_IMAGE); \
        fi

cleanup_remote:
	gsutil rm $(CORE_BUCKET_LOC)
	gcloud container images delete -q --force-delete-tags $(FV3CORE_IMAGE)
