import abc
from typing import List, Optional, TypeVar


T = TypeVar("T")


class Request(abc.ABC):
    @abc.abstractmethod
    def wait(self):
        ...


class Comm(abc.ABC):
    @abc.abstractmethod
    def Get_rank(self) -> int:
        ...

    @abc.abstractmethod
    def Get_size(self) -> int:
        ...

    @abc.abstractmethod
    def bcast(self, value: Optional[T], root=0) -> T:
        ...

    @abc.abstractmethod
    def barrier(self):
        ...

    @abc.abstractmethod
    def Barrier(self):
        ...

    @abc.abstractmethod
    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs):
        ...

    @abc.abstractmethod
    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        ...

    @abc.abstractmethod
    def allgather(self, sendobj: T) -> List[T]:
        ...

    @abc.abstractmethod
    def Send(self, sendbuf, dest, tag: int = 0, **kwargs):
        ...

    @abc.abstractmethod
    def sendrecv(self, sendbuf, dest, **kwargs):
        ...

    @abc.abstractmethod
    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs) -> Request:
        ...

    @abc.abstractmethod
    def Recv(self, recvbuf, source, tag: int = 0, **kwargs):
        ...

    @abc.abstractmethod
    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs) -> Request:
        ...

    @abc.abstractmethod
    def Split(self, color, key) -> "Comm":
        ...

    @abc.abstractmethod
    def allreduce(self, sendobj: T, op=None) -> T:
        ...
