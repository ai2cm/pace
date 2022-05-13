MPIRUN_CALL ?= mpirun -n 6

test:
	pytest tests/main

test_mpi:
	export MPIRUN_CALL
	pytest tests/mpi
	$(MPIRUN_CALL) python3 -m mpi4py -m pace.driver.run examples/configs/baroclinic_c12.yaml
	cd examples && MPIRUN_CALL="$(MPIRUN_CALL)" ./write_then_read.sh

clean:
	rm -rf *.json
	rm -rf RESTART
	rm -rf *.yaml
	rm -rf *output.zarr
