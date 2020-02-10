GCR_URL = us.gcr.io/vcm-ml
TAG ?= latest
FV3_IMAGE=$(GCR_URL)/fv3ser:$(TAG)
TEST_DATA_BUCKET= gs://vcm-fv3gfs-data/serialized-unit-test-data
TEST_DATA_HOST=$(shell pwd)/test_data
TEST_DATA_CONTAINER=/test_data

build:
	cd external/fv3gfs-fortran && make build_environment_serialize
	docker build \
		-f docker/Dockerfile \
		-t $(FV3_IMAGE) \
    .
test_data:
	FORTRAN_COMMIT=git --git-dir=external/fv3gfs-fortran/.git rev-parse HEAD

dev:
	docker run --rm -v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER) -v $(shell pwd):/port_dev -it $(FV3_IMAGE)

tests:
	mkdir -p $(TEST_DATA_HOST)
	# gsutil -m rsync $(TEST_DATA_BUCKET) $(TEST_DATA_HOST) 
	docker run --rm \
	-v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER) \
        -it $(FV3_IMAGE) pytest -v -s  --data_path=$(TEST_DATA_CONTAINER) ${TEST_ARGS} /fv3/test
.PHONY: build tests test_data dev
