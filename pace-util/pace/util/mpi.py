try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import logging
from typing import Optional, TypeVar, cast

from .comm import Comm, Request


T = TypeVar("T")

logger = logging.getLogger(__name__)


class MPIComm(Comm):
    def __init__(self):
        if MPI is None:
            raise RuntimeError("MPI not available")
        self._comm: Comm = cast(Comm, MPI.COMM_WORLD)

    def Get_rank(self) -> int:
        return self._comm.Get_rank()

    def Get_size(self) -> int:
        return self._comm.Get_size()

    def bcast(self, value: Optional[T], root=0) -> T:
        logger.debug("bcast from root %s on rank %s", root, self._comm.Get_rank())
        return self._comm.bcast(value, root=root)

    def barrier(self):
        logger.debug("barrier on rank %s", self._comm.Get_rank())
        self._comm.barrier()

    def Barrier(self):
        pass

    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs):
        logger.debug("Scatter on rank %s with root %s", self._comm.Get_rank(), root)
        self._comm.Scatter(sendbuf, recvbuf, root=root, **kwargs)

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        logger.debug("Gather on rank %s with root %s", self._comm.Get_rank(), root)
        self._comm.Gather(sendbuf, recvbuf, root=root, **kwargs)

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs):
        logger.debug("Send on rank %s with dest %s", self._comm.Get_rank(), dest)
        self._comm.Send(sendbuf, dest, tag=tag, **kwargs)

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs) -> Request:
        logger.debug("Isend on rank %s with dest %s", self._comm.Get_rank(), dest)
        return self._comm.Isend(sendbuf, dest, tag=tag, **kwargs)

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs):
        logger.debug("Recv on rank %s with source %s", self._comm.Get_rank(), source)
        self._comm.Recv(recvbuf, source, tag=tag, **kwargs)

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs) -> Request:
        logger.debug("Irecv on rank %s with source %s", self._comm.Get_rank(), source)
        return self._comm.Irecv(recvbuf, source, tag=tag, **kwargs)

    def Split(self, color, key) -> "Comm":
        logger.debug(
            "Split on rank %s with color %s, key %s", self._comm.Get_rank(), color, key
        )
        return self._comm.Split(color, key)
