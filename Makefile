PYTEST_ARGS?=

test_util: ## run tests quickly with the default Python
	pytest $(PYTEST_ARGS) fv3gfs-util/tests