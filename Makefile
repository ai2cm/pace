GCR_URL = us.gcr.io/vcm-ml
PYTAG ?= latest
FORTRAN_TAG ?= serialize
#<serialization statement change>.<compile options configuration number>.<some other versioned change>
FORTRAN_VERSION=0.0.0
VOLUMES ?=
CWD=$(shell pwd)

TEST_DATA_HOST ?=$(shell pwd)/test_data
TEST_DATA_CONTAINER=/test_data
TEST_DATA_TARGET=fv3gfs-serialization-test-data

SERIALBOX_TARGET=fv3gfs-environment-serialbox
FV3_TARGET ?=fv3ser
FORTRAN=$(CWD)/external/fv3gfs-fortran

FV3_IMAGE ?=$(GCR_URL)/fv3py:$(PYTAG)
COMPILED_IMAGE=$(GCR_URL)/fv3gfs-compiled:$(FORTRAN_VERSION)-$(FORTRAN_TAG)
SERIALBOX_IMAGE=$(GCR_URL)/$(SERIALBOX_TARGET):latest
RUNDIR_IMAGE=$(GCR_URL)/fv3gfs-rundir:$(FORTRAN_VERSION)
RUN_TARGET ?=rundir
TEST_DATA_REPO=$(GCR_URL)/$(TEST_DATA_TARGET)
TEST_DATA_IMAGE=$(TEST_DATA_REPO):$(FORTRAN_VERSION)
TEST_IMAGE=$(GCR_URL)/fv3py-test:$(FORTRAN_VERSION)

FORTRAN_SHA=$(shell git --git-dir=$(FORTRAN)/.git rev-parse HEAD)
FORTRAN_SHA_FILE=fortran_sha.txt
REMOTE_TAGS="$(shell gcloud container images list-tags --format='get(tags)' $(TEST_DATA_REPO) | grep $(FORTRAN_VERSION))"
build_environment_serialize:
	cd $(FORTRAN) && \
	DOCKERFILE=$(FORTRAN)/docker/Dockerfile \
	ENVIRONMENT_TARGET=$(SERIALBOX_TARGET) \
	$(MAKE) build_environment

build: build_environment_serialize
	docker pull $(FV3_IMAGE)
	DOCKER_BUILDKIT=1 docker build \
		--build-arg serialbox_image=$(SERIALBOX_IMAGE) \
		-f docker/Dockerfile \
		-t $(FV3_IMAGE) \
		--target $(FV3_TARGET) \
    .

dev:
	docker run --rm -v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER) -v $(CWD):/port_dev -it $(FV3_IMAGE)

rundir:
	cd $(FORTRAN) && DOCKER_BUILDKIT=1 SERIALIZE_IMAGE=$(COMPILED_IMAGE) $(MAKE) build_serialize
	docker build \
		--build-arg model_image=$(COMPILED_IMAGE) \
		--build-arg fortran_sha_file=$(FORTRAN_SHA_FILE) \
		-f docker/Dockerfile.rundir \
		--target $(DATA_TARGET) \
		-t $(DATA_IMAGE) \
	.

generate_test_data:
	DATA_IMAGE=$(RUNDIR_IMAGE) DATA_TARGET=rundir $(MAKE) rundir
	DATA_IMAGE=$(TEST_DATA_IMAGE) DATA_TARGET=test_data_storage $(MAKE) rundir
	# docker rmi $(RUNDIR_IMAGE)


extract_test_data:
	if [ -d $(TEST_DATA_HOST) ]; then (echo "NOTE: $(TEST_DATA_HOST) already exists, move or delete it if you want a new extraction");\
	else	\
	docker create --name tmp_modelrundata -it $(TEST_DATA_IMAGE)  &&\
	docker cp tmp_modelrundata:/test_data $(TEST_DATA_HOST)  && \
	docker rm -f tmp_modelrundata \
	;fi


post_test_data:
	echo $(REMOTE_TAGS)
	if [ -z $(REMOTE_TAGS) ]; then docker push $(TEST_DATA_IMAGE) ;\
	else echo "ERROR: $(FORTRAN_VERSION) of test data has already been pushed. Do a direct docker push if you really intend to overwrite it" && exit 1 ; fi


pull_test_data:
	[ -z $(docker images -q $(TEST_DATA_IMAGE)) ] || docker pull $(TEST_DATA_IMAGE)

build_tests:
	 DOCKER_BUILDKIT=1 docker build \
		--build-arg fv3ser_image=$(FV3_IMAGE) \
		--build-arg testdata_image=$(TEST_DATA_IMAGE) \
		-f docker/Dockerfile.test \
		--target test \
		-t $(TEST_IMAGE) \
	.

tests: build
	$(MAKE) pull_test_data
	$(MAKE) build_tests
	$(MAKE) run_tests_container


tests_host:
	$(MAKE) pull_test_data
	$(MAKE) extract_test_data
	$(MAKE) run_tests_host_data


test_base:
	docker run --rm $(VOLUMES)\
        -it $(RUNTEST_IMAGE) pytest -v -s  --data_path=$(TEST_DATA_CONTAINER) ${TEST_ARGS} /fv3/test

run_tests_container: 
	RUNTEST_IMAGE=$(TEST_IMAGE) $(MAKE) test_base

run_tests_host_data: 
	VOLUMES='-v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER)' \
	RUNTEST_IMAGE=$(FV3_IMAGE) \
	$(MAKE) test_base

.PHONY: build tests test_data dev
