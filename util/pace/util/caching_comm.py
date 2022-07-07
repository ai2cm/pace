import copy
import dataclasses
import pickle
from typing import Any, BinaryIO, List, Optional, TypeVar

import numpy as np

from .comm import Comm, Request


T = TypeVar("T")


class CachingRequestWriter(Request):
    def __init__(self, req: Request, buffer: np.ndarray, buffer_list: List[np.ndarray]):
        self._req = req
        self._buffer = buffer
        self._buffer_list = buffer_list

    def wait(self):
        self._req.wait()
        self._buffer_list.append(copy.deepcopy(self._buffer))


class CachingRequestReader(Request):
    def __init__(self, recvbuf, data):
        self._recvbuf = recvbuf
        self._data = data

    def wait(self):
        self._recvbuf[:] = self._data


class NullRequest(Request):
    def wait(self):
        pass


@dataclasses.dataclass
class CachingCommData:
    """
    Data required to restore a CachingCommReader.

    Usually you will not want to initialize this class directly, but instead
    use the CachingCommReader.load method.
    """

    rank: int
    size: int
    bcast_objects: List[Any] = dataclasses.field(default_factory=list)
    received_buffers: List[np.ndarray] = dataclasses.field(default_factory=list)
    split_data: List["CachingCommData"] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self._i_bcast = 0
        self._i_buffers = 0
        self._i_split = 0

    def get_bcast(self):
        return_value = self.bcast_objects[self._i_bcast]
        self._i_bcast += 1
        return return_value

    def get_buffer(self):
        return_value = self.received_buffers[self._i_buffers]
        self._i_buffers += 1
        return return_value

    def get_split(self):
        return_value = self.split_data[self._i_split]
        self._i_split += 1
        return return_value

    def dump(self, file: BinaryIO):
        pickle.dump(self, file)

    @classmethod
    def load(self, file: BinaryIO) -> "CachingCommData":
        return pickle.load(file)


class CachingCommReader(Comm):
    """
    mpi4py Comm-like object which replays stored communications.
    """

    def __init__(self, data: CachingCommData):
        """
        Initialize a CachingCommReader.

        Usually you will not want to initialize this class directly, but instead
        use the CachingCommReader.load method.

        Args:
            data: contains all data needed for mocked communication
        """
        self._data = data

    def Get_rank(self) -> int:
        return self._data.rank

    def Get_size(self) -> int:
        return self._data.size

    def bcast(self, value: Optional[T], root=0) -> T:
        return self._data.get_bcast()

    def barrier(self):
        pass

    def Barrier(self):
        pass

    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs):
        recvbuf[:] = self._data.get_buffer()

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        if recvbuf is not None:
            recvbuf[:] = self._data.get_buffer()

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs):
        pass

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs) -> Request:
        return NullRequest()

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs):
        recvbuf[:] = self._data.get_buffer()

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs) -> Request:
        return CachingRequestReader(recvbuf, self._data.get_buffer())

    def Split(self, color, key) -> "CachingCommReader":
        new_data = self._data.get_split()
        return CachingCommReader(data=new_data)

    @classmethod
    def load(cls, file: BinaryIO) -> "CachingCommReader":
        data = CachingCommData.load(file)
        return cls(data)


class CachingCommWriter(Comm):
    """
    Wrapper around a mpi4py Comm object which can be serialized and then loaded
    as a CachingCommReader.
    """

    def __init__(self, comm: Comm):
        """
        Args:
            comm: underlying mpi4py comm-like object
        """
        self._comm = comm
        self._data = CachingCommData(
            rank=comm.Get_rank(),
            size=comm.Get_size(),
        )

    def Get_rank(self) -> int:
        return self._comm.Get_rank()

    def Get_size(self) -> int:
        return self._comm.Get_size()

    def bcast(self, value: Optional[T], root=0) -> T:
        result = self._comm.bcast(value=value, root=root)
        self._data.bcast_objects.append(copy.deepcopy(result))
        return result

    def barrier(self):
        return self._comm.barrier()

    def Barrier(self):
        pass

    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs):
        self._comm.Scatter(sendbuf=sendbuf, recvbuf=recvbuf, root=root, **kwargs)
        self._data.received_buffers.append(copy.deepcopy(recvbuf))

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        self._comm.Gather(sendbuf=sendbuf, recvbuf=recvbuf, root=root, **kwargs)
        self._data.received_buffers.append(copy.deepcopy(recvbuf))

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs):
        self._comm.Send(sendbuf=sendbuf, dest=dest, tag=tag, **kwargs)

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs) -> Request:
        return self._comm.Isend(sendbuf, dest, tag=tag, **kwargs)

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs):
        self._comm.Recv(recvbuf=recvbuf, source=source, tag=tag, **kwargs)
        self._data.received_buffers.append(copy.deepcopy(recvbuf))

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs) -> Request:
        req = self._comm.Irecv(recvbuf, source, tag=tag, **kwargs)
        return CachingRequestWriter(
            req=req, buffer=recvbuf, buffer_list=self._data.received_buffers
        )

    def Split(self, color, key) -> "CachingCommWriter":
        new_comm = self._comm.Split(color=color, key=key)
        new_wrapper = CachingCommWriter(new_comm)
        self._data.split_data.append(new_wrapper._data)
        return new_wrapper

    def dump(self, file: BinaryIO):
        self._data.dump(file)
