import collections
import collections.abc
import functools
import inspect
import types
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import gt4py
import gt4py.definitions
from gt4py import gtscript
from gt4py.storage.storage import Storage

import fv3core
import fv3core._config as spec
import fv3core.utils
import fv3core.utils.global_config as global_config
import fv3core.utils.grid
from fv3core.utils.global_config import StencilConfig
from fv3core.utils.typing import Index3D


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def state_inputs(*arg_specs):
    for sp in arg_specs:
        if sp.intent not in VALID_INTENTS:
            raise ValueError(
                f"intent for {sp.arg_name} is {sp.intent}, "
                "must be one of {VALID_INTENTS}"
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapped(state, *args, **kwargs):
            namespace = get_namespace(arg_specs, state)
            func(namespace, *args, **kwargs)

        return wrapped

    return decorator


def get_namespace(arg_specs, state):
    namespace_kwargs = {}
    for sp in arg_specs:
        arg_name, standard_name, units, intent = sp
        if standard_name not in state:
            raise ValueError(f"{standard_name} not present in state")
        elif units != state[standard_name].units:
            raise ValueError(
                f"{standard_name} has units "
                f"{state[standard_name].units} when {units} is required"
            )
        elif intent not in VALID_INTENTS:
            raise ValueError(
                f"expected intent to be one of {VALID_INTENTS}, got {intent}"
            )
        else:
            namespace_kwargs[arg_name] = state[standard_name].storage
            namespace_kwargs[arg_name + "_quantity"] = state[standard_name]
    return types.SimpleNamespace(**namespace_kwargs)


class FrozenStencil:
    """
    Wrapper for gt4py stencils which stores origin and domain at compile time,
    and uses their stored values at call time.

    This is useful when the stencil itself is meant to be used on a certain
    grid, for example if a compile-time external variable is tied to the
    values of origin and domain.
    """

    def __init__(
        self,
        func: Callable[..., None],
        origin: Union[Index3D, Mapping[str, Tuple[int, ...]]],
        domain: Index3D,
        stencil_config: Optional[StencilConfig] = None,
        externals: Optional[Mapping[str, Any]] = None,
    ):
        """
        Args:
            func: stencil definition function
            origin: gt4py origin to use at call time
            domain: gt4py domain to use at call time
            stencil_config: container for stencil configuration
            externals: compile-time external variables required by stencil
        """
        self.origin = origin

        self.domain = domain

        if stencil_config is not None:
            self.stencil_config: StencilConfig = stencil_config
        else:
            self.stencil_config = global_config.get_stencil_config()

        if externals is None:
            externals = {}

        self.stencil_object: gt4py.StencilObject = gtscript.stencil(
            definition=func,
            externals=externals,
            **self.stencil_config.stencil_kwargs,
        )
        """generated stencil object returned from gt4py."""

        self._argument_names = tuple(inspect.getfullargspec(func).args)

        assert (
            len(self._argument_names) > 0
        ), "A stencil with no arguments? You may be double decorating"

        self._field_origins: Dict[str, Tuple[int, ...]] = compute_field_origins(
            self.stencil_object.field_info, self.origin
        )
        """mapping from field names to field origins"""

        self._stencil_run_kwargs = {
            "_origin_": self._field_origins,
            "_domain_": self.domain,
        }

        self._written_fields = get_written_fields(self.stencil_object.field_info)

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> None:
        if self.stencil_config.validate_args:
            if __debug__ and "origin" in kwargs:
                raise TypeError("origin cannot be passed to FrozenStencil call")
            if __debug__ and "domain" in kwargs:
                raise TypeError("domain cannot be passed to FrozenStencil call")
            self.stencil_object(
                *args,
                **kwargs,
                origin=self._field_origins,
                domain=self.domain,
                validate_args=True,
            )
        else:
            args_as_kwargs = dict(zip(self._argument_names, args))
            self.stencil_object.run(
                **args_as_kwargs, **kwargs, **self._stencil_run_kwargs, exec_info=None
            )
            self._mark_cuda_fields_written({**args_as_kwargs, **kwargs})

    def _mark_cuda_fields_written(self, fields: Mapping[str, Storage]):
        if "cuda" in self.stencil_config.backend:
            for write_field in self._written_fields:
                fields[write_field]._set_device_modified()


def get_stencils_with_varied_bounds(
    func: Callable[..., None],
    origins: List[Index3D],
    domains: List[Index3D],
    stencil_config: Optional[StencilConfig] = None,
    externals: Optional[Mapping[str, Any]] = None,
) -> List[FrozenStencil]:
    assert len(origins) == len(domains), (
        "Lists of origins and domains need to have the same length, you provided "
        + str(len(origins))
        + " origins and "
        + str(len(domains))
        + " domains"
    )
    if externals is None:
        externals = {}
    stencils = []
    for origin, domain in zip(origins, domains):
        ax_offsets = fv3core.utils.grid.axis_offsets(spec.grid, origin, domain)
        stencils.append(
            FrozenStencil(
                func,
                origin=origin,
                domain=domain,
                stencil_config=stencil_config,
                externals={**externals, **ax_offsets},
            )
        )
    return stencils


def get_written_fields(field_info) -> List[str]:
    """Returns the list of fields that are written.

    Args:
        field_info: field_info attribute of gt4py stencil object
    """
    write_fields = [
        field_name
        for field_name in field_info
        if field_info[field_name]
        and field_info[field_name].access != gt4py.definitions.AccessKind.READ_ONLY
    ]
    return write_fields


def compute_field_origins(
    field_info_mapping, origin: Union[Index3D, Mapping[str, Tuple[int, ...]]]
) -> Dict[str, Tuple[int, ...]]:
    """
    Computes the origin for each field in the stencil call.

    Args:
        field_info_mapping: from stencil.field_info, a mapping which gives the
            dimensionality of each input field
        origin: the (i, j, k) coordinate of the origin

    Returns:
        origin_mapping: a mapping from field names to origins
    """
    if isinstance(origin, tuple):
        field_origins: Dict[str, Tuple[int, ...]] = {"_all_": origin}
        origin_tuple: Tuple[int, ...] = origin
    else:
        field_origins = {**origin}
        origin_tuple = origin["_all_"]
    field_names = tuple(field_info_mapping.keys())
    for i, field_name in enumerate(field_names):
        if field_name not in field_origins:
            field_info = field_info_mapping[field_name]
            if field_info is not None:
                field_origin_list = []
                for ax in field_info.axes:
                    origin_index = {"I": 0, "J": 1, "K": 2}[ax]
                    field_origin_list.append(origin_tuple[origin_index])
                field_origin = tuple(field_origin_list)
            else:
                field_origin = origin_tuple
            field_origins[field_name] = field_origin
    return field_origins


def gtstencil(
    func,
    origin: Optional[Index3D] = None,
    domain: Optional[Index3D] = None,
    stencil_config: Optional[StencilConfig] = None,
    externals: Optional[Mapping[str, Any]] = None,
):
    """
    Returns a wrapper over gt4py stencils.

    If origin and domain are not given, they must be provided at call time,
    and a separate stencil is compiled for each origin and domain pair used.

    Args:
        func: stencil definition function
        origin: the start of the compute domain
        domain: the size of the compute domain, required if origin is given
        stencil_config: stencil configuration, by default global stencil
            configuration at the first call time is used
        externals: compile-time constants used by stencil

    Returns:
        wrapped_stencil: an object similar to gt4py stencils, takes origin
            and domain as arguments if and only if they were not given
            as arguments to gtstencil
    """
    if not (origin is None) == (domain is None):
        raise TypeError("must give both origin and domain arguments, or neither")
    if externals is None:
        externals = {}
    if origin is None:
        stencil = get_non_frozen_stencil(func, externals)
    else:
        # TODO: delete this global default
        if stencil_config is None:
            stencil_config = global_config.get_stencil_config()
        stencil = FrozenStencil(
            func,
            origin=origin,
            domain=domain,
            stencil_config=stencil_config,
            externals=externals,
        )
    return stencil


def get_non_frozen_stencil(func, externals) -> Callable[..., None]:
    stencil_dict: Dict[Hashable, FrozenStencil] = {}
    # must use a mutable container object here to hold config,
    # `global` does not work in this case. Cannot retreve StencilConfig
    # yet because it is not set at import time, when this function
    # is called by gtstencil throughout the repo
    # so instead we retrieve it at first call time
    stencil_config_holder: List[StencilConfig] = []

    @functools.wraps(func)
    def decorated(
        *args,
        origin: Union[Index3D, Mapping[str, Tuple[int, ...]]],
        domain: Index3D,
        **kwargs,
    ):
        try:
            stencil_config = stencil_config_holder[0]
        except IndexError:
            stencil_config = global_config.get_stencil_config()
            stencil_config_holder.append(stencil_config)
        try:  # works if origin is a Mapping
            origin_key: Hashable = tuple(
                sorted(origin.items(), key=lambda x: x[0])  # type: ignore
            )
            origin_tuple: Tuple[int, ...] = origin["_all_"]  # type: ignore
        except AttributeError:  # assume origin is a tuple
            origin_key = origin
            origin_tuple = cast(Index3D, origin)
        # rank is needed in the key for regression testing
        # for > 6 ranks, where each rank may or may not be
        # on a tile edge
        key: Hashable = (origin_key, domain, spec.grid.rank)
        if key not in stencil_dict:
            axis_offsets = fv3core.utils.grid.axis_offsets(
                spec.grid, origin=origin_tuple, domain=domain
            )
            stencil_dict[key] = FrozenStencil(
                func,
                origin,
                domain,
                stencil_config=stencil_config,
                externals={**axis_offsets, **externals, "namelist": spec.namelist},
            )
        return stencil_dict[key](*args, **kwargs)

    return decorated
