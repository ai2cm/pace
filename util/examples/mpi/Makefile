
MPI_ENV_VARS=PMIX_MCA_gds=hash

all: global_timings zarr_monitor

global_timings:
	$(MPI_ENV_VARS) mpirun -n 4 python -m mpi4py global_timings.py

zarr_monitor:
	$(MPI_ENV_VARS) mpirun -n 6 python -m mpi4py zarr_monitor.py

clean:
	$(RM) -r output/*
	touch output/.gitkeep
