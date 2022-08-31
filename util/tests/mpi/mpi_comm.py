try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI is not None and MPI.COMM_WORLD.Get_size() == 1:
    # not run as a parallel test, disable MPI tests
    MPI.Finalize()
    MPI = None
