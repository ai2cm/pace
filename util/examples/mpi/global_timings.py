import contextlib

import numpy as np
from mpi4py import MPI

from pace.util import Timer


@contextlib.contextmanager
def nullcontext():
    yield


def print_global_timings(times, comm, root=0):
    is_root = comm.Get_rank() == root
    recvbuf = np.array(0.0)
    for name, value in timer.times.items():
        if is_root:
            print(name)
        for label, op in [("min", MPI.MIN), ("max", MPI.MAX), ("mean", MPI.SUM)]:
            comm.Reduce(np.array(value), recvbuf, op=op)
            if is_root:
                if label == "mean":
                    recvbuf /= comm.Get_size()
                print(f"    {label}: {recvbuf}")


if __name__ == "__main__":
    # a Timer gathers statistics about the blocks it times
    arr = np.random.randn(100, 100)
    timer = Timer()

    # using a context manager ensures that stop is always called, even if there is an
    # exception/error in the block. We strongly encourage using this method when
    # possible.
    with timer.clock("addition"):
        arr += 1

    # sometimes, you will need to trigger the start and end of the timer manually, if
    # the start and end cannot be represented by a context manager
    timer.start("context_manager")
    with nullcontext():
        timer.stop("context_manager")

    comm = MPI.COMM_WORLD
    # timer.times is a dictionary giving you the total time in seconds spent on each
    # operation
    print_global_timings(timer.times, comm)
