GCR_URL = us.gcr.io/vcm-ml
TAG ?= latest
FV3_IMAGE=$(GCR_URL)/fv3ser:$(TAG)
TEST_DATA_BUCKET= gs://vcm-fv3gfs-data/serialized-unit-test-data
TEST_DATA_HOST=$(shell pwd)/test_data
TEST_DATA_CONTAINER=/test_data
GT4PY_PATH=$(shell pwd)/gridtools4py
GT4PY_REPO=git@github.com:GridTools/gt4py.git
run_serialize:
	rm -f $(RUNDIR_HOST)/Gen*.dat
	rm -f $(RUNDIR_HOST)/Archive*.json
	rm -f $(RUNDIR_HOST)/Meta*.json
	if [ ! -d $(shell pwd)/inputdata/fv3gfs-data-docker/fix.v201702 ];then\
	    ./download_inputdata.sh ;\
	fi
	docker run --rm \
		-v $(RUNDIR_HOST):/Serialize/$(RUNDIR_CONTAINER) \
		-v $(shell pwd)/inputdata/fv3gfs-data-docker/fix.v201702:/inputdata/fix.v201702 \
		-it $(GCR_URL)/fv3gfs-compiled-serialize /Serialize/FV3/rundir/submit_job.sh /Serialize/FV3
build:
	if [ ! -d "$(GT4PY_PATH)" ];then \
		git clone $(GT4PY_REPO) $(GT4PY_PATH);\
	fi
	docker build \
		-f docker/Dockerfile \
		-t $(FV3_IMAGE) \
    .
tests:
	mkdir -p $(TEST_DATA_HOST)
	gsutil -m rsync $(TEST_DATA_BUCKET) $(TEST_DATA_HOST) 
	docker run --rm \
	-v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER) \
        -it $(FV3_IMAGE) pytest -v -s  --data_path=$(TEST_DATA_CONTAINER) ${TEST_ARGS} /fv3/test
.PHONY: build build_environment build_compiled enter run
