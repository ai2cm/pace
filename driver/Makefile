MPIRUN_CALL ?= mpirun -n 6

test:
	pytest tests --ignore=tests/test_restart.py

test_mpi:
	$(MPIRUN_CALL) python3 -m mpi4py -m pace.driver.run examples/configs/baroclinic_c12.yaml
	cd examples && MPIRUN_CALL="$(MPIRUN_CALL)" ./write_then_read.sh

test_restart:
	cp examples/configs/baroclinic_c12_write_restart.yaml baroclinic_c12_write_restart.yaml
	cp examples/configs/baroclinic_c12_write_restart.yaml baroclinic_c12_run_two_steps.yaml
	sed -i 's/seconds: 225/seconds: 450/' baroclinic_c12_run_two_steps.yaml
	sed -i 's/save_restart: true/save_restart: false/' baroclinic_c12_run_two_steps.yaml
	sed -i 's/path: "output.zarr"/path: "run_two_steps_output.zarr"/' baroclinic_c12_run_two_steps.yaml
	$(MPIRUN_CALL) -n 6 python3 -m pace.driver.run baroclinic_c12_write_restart.yaml
	$(MPIRUN_CALL) -n 6 python3 -m pace.driver.run RESTART/restart.yaml
	$(MPIRUN_CALL) -n 6 python3 -m pace.driver.run baroclinic_c12_run_two_steps.yaml
	pytest tests -k test_restart

clean:
	rm -rf *.json
	rm -rf RESTART
	rm -rf *.yaml
	rm -rf *output.zarr
