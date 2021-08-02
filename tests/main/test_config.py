import collections
import dataclasses
import typing

import pytest

import fv3core
import fv3core._config


CONFIG_CLASSES = [
    fv3core._config.AcousticDynamicsConfig,
    fv3core._config.RiemannConfig,
    fv3core._config.DGridShallowWaterLagrangianDynamicsConfig,
    fv3core._config.Namelist,
]


def test_set_backend():
    start_backend = fv3core.get_backend()
    new_backend = "new_backend"
    assert new_backend != start_backend
    fv3core.set_backend(new_backend)
    assert fv3core.get_backend() == new_backend


@dataclasses.dataclass
class FirstConfigClass:
    value: float


@dataclasses.dataclass
class CompatibleConfigClass:
    value: float


@dataclasses.dataclass
class IncompatibleConfigClass:
    value: int


@dataclasses.dataclass
class IncompatiblePropertyConfigClass:
    @property
    def value(self) -> int:
        return 0


def assert_types_match(classes):
    types = collections.defaultdict(set)
    for cls in classes:
        for name, field in cls.__dataclass_fields__.items():
            types[name].add(field.type)
        for name, attr in cls.__dict__.items():
            if isinstance(attr, property):
                types[name].add(
                    typing.get_type_hints(attr.fget).get("return", typing.Any)
                )
    assert not any(len(type_list) > 1 for type_list in types.values()), {
        key: value for key, value in types.items() if len(value) > 1
    }


def test_assert_types_match_compatible_types():
    assert_types_match([FirstConfigClass, CompatibleConfigClass])


def test_assert_types_match_incompatible_types():
    with pytest.raises(AssertionError):
        assert_types_match([FirstConfigClass, IncompatibleConfigClass])


def test_assert_types_match_incompatible_property_type():
    with pytest.raises(AssertionError):
        assert_types_match([FirstConfigClass, IncompatiblePropertyConfigClass])


def test_types_match():
    """
    Test that when an attribute exists on two or more configuration dataclasses,
    their type hints are the same.

    Checks both dataclass attributes and property methods.
    """
    assert_types_match(CONFIG_CLASSES)
