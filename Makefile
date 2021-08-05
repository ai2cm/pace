test_util:
	$(MAKE) -C fv3gfs-util test

test_dycore_serial:
	$(MAKE) -C fv3core savepoint_tests

test_dycore_parallel:
	$(MAKE) -C fv3core savepoint_tests_mpi
