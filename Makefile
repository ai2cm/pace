GCR_URL = us.gcr.io/vcm-ml
CWD=$(shell pwd)
#<some large conceptual version change>.<serialization statement change>.<hotfix>
FORTRAN_VERSION=0.1.0


PULL ?=True
VOLUMES ?=
TEST_DATA_HOST ?=$(CWD)/test_data
FV3_IMAGE ?=$(GCR_URL)/fv3ser:latest

FV3_INSTALL_TARGET=fv3ser-install
FV3_INSTALL_IMAGE=$(GCR_URL)/$(FV3_INSTALL_TARGET):latest

FORTRAN_DIR=$(CWD)/external/fv3gfs-fortran
COMPILED_IMAGE=$(GCR_URL)/fv3gfs-compiled:$(FORTRAN_VERSION)-serialize
SERIALBOX_TARGET=fv3gfs-environment-serialbox
SERIALBOX_IMAGE=$(GCR_URL)/$(SERIALBOX_TARGET):latest
BASE_ENV_IMAGE=$(GCR_URL)/fv3gfs-environment:latest
RUNDIR_IMAGE=$(GCR_URL)/fv3gfs-rundir:$(FORTRAN_VERSION)

TEST_DATA_CONTAINER=/test_data
TEST_DATA_REPO=$(GCR_URL)/fv3gfs-serialization-test-data
TEST_DATA_IMAGE=$(TEST_DATA_REPO):$(FORTRAN_VERSION)
TEST_IMAGE=$(GCR_URL)/fv3ser-test:$(FORTRAN_VERSION)
TEST_DATA_RUN_CONTAINER=TestDataContainer-$(FORTRAN_VERSION)

FORTRAN_SHA=$(shell git --git-dir=$(FORTRAN_DIR)/.git rev-parse HEAD)
REMOTE_TAGS="$(shell gcloud container images list-tags --format='get(tags)' $(TEST_DATA_REPO) | grep $(FORTRAN_VERSION))"

PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')

build_environment_serialize:
	if [ ! -d $(FORTRAN_DIR)/FV3 ]; then git submodule update --init --recursive ;fi
	cd $(FORTRAN_DIR) && \
	DOCKERFILE=$(FORTRAN_DIR)/docker/Dockerfile \
	ENVIRONMENT_TARGET=$(SERIALBOX_TARGET) \
	$(MAKE) build_environment

build_environment: build_environment_serialize
	DOCKER_BUILDKIT=1 docker build \
		--build-arg serialbox_image=$(SERIALBOX_IMAGE) \
		-f docker/Dockerfile.build_environment \
		-t $(FV3_INSTALL_IMAGE) \
	--target $(FV3_INSTALL_TARGET) \
    .

build:
	if [ $(PULL) == True ]; then $(MAKE) pull_environment; else $(MAKE) build_environment; fi
	docker build --build-arg build_image=$(FV3_INSTALL_IMAGE) -f docker/Dockerfile -t $(FV3_IMAGE) .

pull_environment:
	if [ -z $(shell docker images -q $(FV3_INSTALL_IMAGE)) ]; then docker pull $(FV3_INSTALL_IMAGE) ;fi

push_environment:
	docker push $(FV3_INSTALL_IMAGE)

rebuild_environment: build_environment
	$(MAKE) push_environment

dev:
	docker run --rm -v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER) -v $(CWD):/port_dev -it $(FV3_IMAGE)

devc:
	if [ -z $(shell docker ps -q -f name=$(TEST_DATA_RUN_CONTAINER)) ]; then $(MAKE) data_container;fi
	docker run --rm --volumes-from $(TEST_DATA_RUN_CONTAINER) -v $(CWD):/port_dev -it $(FV3_IMAGE)

fortran_model_data: #uses the 'fv3config.yml' in the fv3gfs-fortran regression tests to configure a test run for generation serialization data
	docker build \
		--build-arg model_image=$(COMPILED_IMAGE) \
		--build-arg commit_hash=$(FORTRAN_SHA)\
		-f docker/Dockerfile.fortran_model_data \
		--target $(DATA_TARGET) \
		-t $(DATA_IMAGE) \
	.

generate_test_data:
	if [ ! -d $(FORTRAN_DIR)/FV3 ]; then git submodule update --init --recursive ;fi
	cd $(FORTRAN_DIR) && DOCKER_BUILDKIT=1 SERIALIZE_IMAGE=$(COMPILED_IMAGE) $(MAKE) build_serialize
	DATA_IMAGE=$(RUNDIR_IMAGE) DATA_TARGET=rundir $(MAKE) fortran_model_data
	DATA_IMAGE=$(TEST_DATA_IMAGE) DATA_TARGET=test_data_storage $(MAKE) fortran_model_data
	docker rmi $(RUNDIR_IMAGE)


extract_test_data:
	if [ -d $(TEST_DATA_HOST) ]; then (echo "NOTE: $(TEST_DATA_HOST) already exists, move or delete it if you want a new extraction");\
	else	\
	docker create --name tmp_modelrundata -it $(TEST_DATA_IMAGE)  &&\
	docker cp tmp_modelrundata:/test_data $(TEST_DATA_HOST)  && \
	docker rm -f tmp_modelrundata \
	;fi


post_test_data:
	if [ -z $(REMOTE_TAGS) ]; then docker push $(TEST_DATA_IMAGE) ;\
	else echo "ERROR: $(FORTRAN_VERSION) of test data has already been pushed. Do a direct docker push if you really intend to overwrite it" && exit 1 ; fi


pull_test_data:
	docker pull $(TEST_DATA_IMAGE)

tests:
	$(MAKE) build
	if [ -z $(shell docker images -q $(TEST_DATA_IMAGE)) ]; then $(MAKE) pull_test_data ;fi
	if [ -z $(shell docker ps -q -f name=$(TEST_DATA_RUN_CONTAINER)) ]; then $(MAKE) data_container;fi
	$(MAKE) run_tests_container

data_container:
	docker run -d -it --name=$(TEST_DATA_RUN_CONTAINER) -v TestDataVolume$(FORTRAN_VERSION):/test_data $(TEST_DATA_IMAGE)

cleanup_container:
	docker stop $(TEST_DATA_RUN_CONTAINER)
	docker rm $(TEST_DATA_RUN_CONTAINER)

tests_host:
	$(MAKE) pull_test_data
	$(MAKE) extract_test_data
	$(MAKE) run_tests_host_data


test_base:
	docker run --rm $(VOLUMES) \
	-it $(RUNTEST_IMAGE) pytest -v -s  --data_path=$(TEST_DATA_CONTAINER) ${TEST_ARGS} /fv3/test

run_tests_container:
	VOLUMES='--volumes-from $(TEST_DATA_RUN_CONTAINER)' \
	RUNTEST_IMAGE=$(FV3_IMAGE) $(MAKE) test_base

run_tests_host_data: 
	VOLUMES='-v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER)' \
	RUNTEST_IMAGE=$(FV3_IMAGE) \
	$(MAKE) test_base

lint:
	black --diff --check $(PYTHON_FILES) $(PYTHON_INIT_FILES)
	# disable flake8 tests for now, re-enable when dycore is "running"
	#flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	#flake8 --ignore=F401 $(PYTHON_INIT_FILES)
	@echo "LINTING SUCCESSFUL"

flake8:
	flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	flake8 --ignore=F401 $(PYTHON_INIT_FILES)

reformat:
	black $(PYTHON_FILES) $(PYTHON_INIT_FILES)

.PHONY: build tests tests_host test_base run_tests_container run_tests_host_data \
	dev devc generate_test_data extract_test_data post_test_data pull_test_data \
	data_container fortran_model_data pull_environment push_environment  \
	build_environment build_environment_serialize cleanup_container flake8 lint reformat

