from typing import Any, Mapping

from pace.util.comm import Comm, Request


class NullAsyncResult(Request):
    def __init__(self, recvbuf=None):
        self._recvbuf = recvbuf

    def wait(self):
        if self._recvbuf is not None:
            self._recvbuf[:] = 0.0


class NullComm(Comm):
    """
    A class with a subset of the mpi4py Comm API, but which
    'receives' a fill value (default zero) instead of using MPI.
    """

    def __init__(self, rank, total_ranks, fill_value=0.0):
        """
        Args:
            rank: rank to mock
            total_ranks: number of total MPI ranks to mock
            fill_value: fill halos with this value when performing
                halo updates.
        """
        self.rank = rank
        self.total_ranks = total_ranks
        self._fill_value = fill_value
        self._split_comms: Mapping[Any, NullComm] = {}

    def __repr__(self):
        return f"NullComm(rank={self.rank}, total_ranks={self.total_ranks})"

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.total_ranks

    def bcast(self, value, root=0):
        return value

    def barrier(self):
        return

    def Barrier(self):
        return

    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs):
        if recvbuf is not None:
            recvbuf[:] = self._fill_value

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        if recvbuf is not None:
            recvbuf[:] = self._fill_value

    def Send(self, sendbuf, dest, **kwargs):
        pass

    def Isend(self, sendbuf, dest, **kwargs):
        return NullAsyncResult()

    def Recv(self, recvbuf, source, **kwargs):
        recvbuf[:] = self._fill_value

    def Irecv(self, recvbuf, source, **kwargs):
        return NullAsyncResult(recvbuf)

    def Split(self, color, key):
        # key argument is ignored, assumes we're calling the ranks from least to
        # greatest when mocking Split
        self._split_comms[color] = self._split_comms.get(color, [])
        rank = len(self._split_comms[color])
        total_ranks = rank + 1
        new_comm = NullComm(
            rank=rank, total_ranks=total_ranks, fill_value=self._fill_value
        )
        for comm in self._split_comms[color]:
            # won't know how many ranks there are until everything is split
            comm.total_ranks = total_ranks
        self._split_comms[color].append(new_comm)
        return new_comm
