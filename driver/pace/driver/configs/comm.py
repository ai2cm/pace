import abc
import dataclasses
import os
from typing import Any, ClassVar, List

import dacite

import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from pace.util.caching_comm import CachingCommReader, CachingCommWriter


class CreatesComm(abc.ABC):
    @abc.abstractmethod
    def get_comm(self) -> Any:
        """
        Get an mpi4py-style Comm object.
        """
        ...

    @abc.abstractmethod
    def cleanup(self, comm):
        """
        Perform any operations that must occur before exiting.
        """
        ...


@dataclasses.dataclass(frozen=True)
class CommConfig:
    config: CreatesComm
    type: str = "mpi"
    registry: ClassVar = {}

    @classmethod
    def register(cls, type_name):
        def register_func(func):
            cls.registry[type_name] = func
            return func

        return register_func

    def get_comm(self):
        return self.config.get_comm()

    def cleanup(self, comm):
        return self.config.cleanup(comm)

    @classmethod
    def from_dict(cls, config: dict):
        config.setdefault("config", {})
        comm_type = config.get("type", "mpi")
        if comm_type not in cls.registry:
            raise ValueError(f"Unknown comm type: {comm_type}")
        else:
            creates_comm: CreatesComm = dacite.from_dict(
                data_class=cls.registry[comm_type],
                data=config["config"],
                config=dacite.Config(strict=True),
            )
            return cls(config=creates_comm, type=comm_type)


@CommConfig.register("mpi")
@dataclasses.dataclass
class MPICommConfig(CreatesComm):
    def get_comm(self):
        return pace.util.MPIComm()

    def cleanup(self, comm):
        pass


@CommConfig.register("write")
@dataclasses.dataclass
class WriterCommConfig(CreatesComm):
    ranks: List[int]
    path: str = "."

    def get_comm(self) -> CachingCommWriter:
        underlying = MPICommConfig().get_comm()
        if underlying.Get_rank() in self.ranks:
            return pace.util.CachingCommWriter(underlying)
        else:
            return underlying

    def cleanup(self, comm: CachingCommWriter):
        os.makedirs(self.path, exist_ok=True)
        if comm.Get_rank() in self.ranks:
            with open(
                os.path.join(self.path, f"comm_{comm.Get_rank()}.pkl"), "wb"
            ) as f:
                comm.dump(f)


@CommConfig.register("read")
@dataclasses.dataclass
class ReaderCommConfig(CreatesComm):
    rank: int
    path: str = "."

    def get_comm(self) -> CachingCommReader:
        with open(os.path.join(self.path, f"comm_{self.rank}.pkl"), "rb") as f:
            return pace.util.CachingCommReader.load(f)

    def cleanup(self, comm: CachingCommWriter):
        pass
