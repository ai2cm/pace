MPIRUN_CALL ?= mpirun -n 6

test_mpi:
	MPIRUN_CALL="$(MPIRUN_CALL)" pytest tests/mpi
	$(MPIRUN_CALL) python3 -m mpi4py -m pace.driver.run examples/configs/baroclinic_c12.yaml
	cd examples && MPIRUN_CALL="$(MPIRUN_CALL)" ./write_then_read.sh

clean:
	rm -rf *.json
	rm -rf RESTART
	rm -rf baroclinic_c12_*.yaml
	rm -rf *output.zarr
