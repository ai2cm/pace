import dataclasses
from typing import Callable, Dict, Generic, Optional, Type, TypeVar

import dacite


T = TypeVar("T")
TT = TypeVar("TT", bound=Type)


@dataclasses.dataclass
class ConfigSpecification(Generic[T]):
    """
    A section of yaml configuration containing one of multiple possible
    dataclasses, conforming to the generic type specified.

    Examples:

    """

    type: str
    config: T


class Registry(Generic[T]):
    """
    Used to register and initialize multiple types of a dataclass.

    Examples:

        First, let's register a class MyConfig with a type name
        "my_type" and then initialize it. First we import the required symbols:

        >>> import dataclasses
        >>> from pace.driver.registry import Registry

        Then we define a registry and register a class MyConfig:

        >>> registry = Registry()
        >>> @registry.register("my_type")
        ... @dataclasses.dataclass
        ... class MyConfig:
        ...     a: int
        ...     b: int

        Once MyConfig has been registered with the type name "my_type", the
        registry knows to construct it when it sees this type name in a dictionary.
        This is especially useful for selecting a type using yaml configuration:

        >>> registry.from_dict({"type": "my_type", "config": {"a": 1, "b": 2}})
        MyConfig(a=1, b=2)

        If a default type is given when the Registry is created, it will get
        initialized when no "type" key is given. Otherwise, the "type" key is
        required.

        >>> diagnostics_registry = Registry(default_type="no_diagnostics")
        >>> @diagnostics_registry.register("no_diagnostics")
        ... @dataclasses.dataclass
        ... class NoDiagnostics:
        ...     def store_state(self, state):
        ...         pass

        Since the diagnostics registry above has a default type, and the default
        type has no required arguments to initialize it, we can initialize a
        diagnostics class that implements the store_state method without specifying
        any config:

        >>> diagnostics_registry.from_dict({})
        NoDiagnostics()

        The above example could be used to register a default "diagnostics" class
        which does nothing, and can be swapped with others which might calculate
        certain values and store them.
    """

    def __init__(self, default_type: Optional[str] = None):
        """
        Initialize the registry.

        Args:
            default_type: if given, the "type" key in the config dict is optional
                and by default this type will be used.
        """
        self._types: Dict[str, Type[T]] = {}
        self.default_type = default_type

    def register(self, type_name: str) -> Callable[[TT], TT]:
        """
        Registers a configuration type with the registry.

        When registry.from_dict is called to initialize a dataclass, if the
        "type" key in that dictionary is equal to the type_name you give here,
        then the decorated class will be the one initialized from the data
        in the "config" key.

        Args:
            type_name: name used in configuration to indicate the decorated
                class as the target type to be initialized when using from_dict.
        """

        def register_func(cls: TT) -> TT:
            self._types[type_name] = cls
            return cls

        return register_func

    def from_dict(self, config: dict) -> T:
        """
        Creates a registered type from the given config dict.

        Config should have at least one key, "type", which indicates the type to
        initialize based on its registered type name. This can be omitted if
        this instance was initialized with a default type.

        It can also have a "config" key, which is a dict used to initialize the
        dataclass. By default this is an empty dict.
        """
        config.setdefault("config", {})
        if self.default_type is not None:
            type_name = config.get("type", self.default_type)
        else:
            type_name = config["type"]
        if type_name not in self._types:
            raise ValueError(
                f"Received unexpected type {type_name}, "
                f"expected one of {self._types.keys()}"
            )
        else:
            instance = dacite.from_dict(
                data_class=self._types[type_name],
                data=config["config"],
                config=dacite.Config(strict=True),
            )
            return instance
