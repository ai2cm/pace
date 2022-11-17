import abc
import dataclasses
import logging
from typing import ClassVar

import dacite
import yaml

from pace.util import SavepointThresholds
from pace.util.checkpointer import (
    Checkpointer,
    NullCheckpointer,
    SnapshotCheckpointer,
    ThresholdCalibrationCheckpointer,
    ValidationCheckpointer,
)

from .registry import Registry


logger = logging.getLogger(__name__)


class CheckpointerInitializer(abc.ABC):
    @abc.abstractmethod
    def get_checkpointer(self, rank: int) -> Checkpointer:
        ...


@dataclasses.dataclass
class CheckpointerInitializerSelector(CheckpointerInitializer):
    """
    Dataclass for selecting the implementation of CheckpointerInitializer to use.

    Used to circumvent the issue that dacite expects static class definitions,
    but we would like to dynamically define which CheckpointerInitializer to use.
    Does this by representing the part of the yaml specification that asks which
    initializer to use, but deferring to the implementation
    in that initializer when called.
    """

    type: str
    config: CheckpointerInitializer
    registry: ClassVar[Registry] = Registry()

    @classmethod
    def register(cls, type_name):
        return cls.registry.register(type_name)

    def get_checkpointer(self, rank: int) -> Checkpointer:
        return self.config.get_checkpointer(rank=rank)

    @classmethod
    def from_dict(cls, config: dict):
        instance = cls.registry.from_dict(config)
        return cls(config=instance, type=config["type"])


@CheckpointerInitializerSelector.register("null")
@dataclasses.dataclass
class NullCheckpointerInit(CheckpointerInitializer):
    """
    Configuration for threshold calibration checkpointer.
    """

    def get_checkpointer(self, rank: int) -> Checkpointer:
        return NullCheckpointer()


@CheckpointerInitializerSelector.register("threshold_calibration")
@dataclasses.dataclass
class ThresholdCalibrationCheckpointerInit(CheckpointerInitializer):
    """
    Configuration for threshold calibration checkpointer.
    """

    factor: float = 10.0

    def get_checkpointer(self, rank: int) -> Checkpointer:
        return ThresholdCalibrationCheckpointer(self.factor)


@CheckpointerInitializerSelector.register("validation")
@dataclasses.dataclass
class ValidationCheckpointerInit(CheckpointerInitializer):
    """
    Configuration for validation checkpointer.
    """

    savepoint_data_path: str
    threshold_filename: str

    def get_checkpointer(self, rank: int) -> Checkpointer:
        with open(self.threshold_filename, "r") as f:
            data = yaml.safe_load(f)
            thresholds = dacite.from_dict(
                data_class=SavepointThresholds,
                data=data,
                config=dacite.Config(strict=True),
            )
        return ValidationCheckpointer(
            savepoint_data_path=self.savepoint_data_path,
            thresholds=thresholds,
            rank=rank,
        )


@CheckpointerInitializerSelector.register("snapshot")
@dataclasses.dataclass
class SnapshotCheckpointerInit(CheckpointerInitializer):
    """
    Configuration for snapshot checkpointer.
    """

    def get_checkpointer(self, rank: int) -> Checkpointer:
        return SnapshotCheckpointer(rank=rank)
