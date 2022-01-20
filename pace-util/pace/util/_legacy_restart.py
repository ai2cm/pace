import copy
import os
from typing import BinaryIO, Generator, Iterable

from . import _xarray as xr
from . import constants, filesystem, io
from ._properties import RESTART_PROPERTIES, RestartProperties
from .communicator import CubedSphereCommunicator
from .partitioner import get_tile_index
from .quantity import Quantity


__all__ = ["open_restart"]

RESTART_NAMES = ("fv_core.res", "fv_srf_wnd.res", "fv_tracer.res")
RESTART_OPTIONAL_NAMES = ("sfc_data", "phy_data")  # not output for dycore-only runs
COUPLER_RES_NAME = "coupler.res"


def open_restart(
    dirname: str,
    communicator: CubedSphereCommunicator,
    label: str = "",
    only_names: Iterable[str] = None,
    to_state: dict = None,
    tracer_properties: RestartProperties = None,
):
    """Load restart files output by the Fortran model into a state dictionary.

    Args:
        dirname: location of restart files, can be local or remote
        communicator: object for communication over the cubed sphere
        label: prepended string on the restart files to load
        only_names (optional): list of standard names to load
        to_state (optional): if given, assign loaded data into pre-allocated quantities
            in this state dictionary

    Returns:
        state: model state dictionary
    """
    if tracer_properties is None:
        restart_properties = RESTART_PROPERTIES
    else:
        restart_properties = {**tracer_properties, **RESTART_PROPERTIES}
    rank = communicator.rank
    tile_index = communicator.partitioner.tile_index(rank)
    state = {}
    if communicator.tile.rank == constants.ROOT_RANK:
        for file in restart_files(dirname, tile_index, label):
            state.update(
                load_partial_state_from_restart_file(
                    file, restart_properties, only_names=only_names
                )
            )
        coupler_res_filename = get_coupler_res_filename(dirname, label)
        if filesystem.is_file(coupler_res_filename):
            if only_names is None or "time" in only_names:
                with filesystem.open(coupler_res_filename, "r") as f:
                    state["time"] = io.get_current_date_from_coupler_res(f)
    if to_state is None:
        state = communicator.tile.scatter_state(state)
    else:
        state = communicator.tile.scatter_state(state, recv_state=to_state)
    return state


def get_coupler_res_filename(dirname, label):
    return os.path.join(dirname, prepend_label(COUPLER_RES_NAME, label))


def restart_files(dirname, tile_index, label) -> Generator[BinaryIO, None, None]:
    for filename in restart_filenames(dirname, tile_index, label):
        with filesystem.open(filename, "rb") as f:
            yield f


def restart_filenames(dirname, tile_index, label):
    suffix = f".tile{tile_index + 1}.nc"
    return_list = []
    for name in RESTART_NAMES + RESTART_OPTIONAL_NAMES:
        filename = os.path.join(dirname, prepend_label(name, label) + suffix)
        if (
            (name in RESTART_NAMES)
            or filesystem.is_file(filename)
            or os.path.exists(filename)
        ):
            return_list.append(filename)
    return return_list


def get_rank_suffix(rank, total_ranks):
    if total_ranks % 6 != 0:
        raise ValueError(
            f"total_ranks must be evenly divisible by 6, was given {total_ranks}"
        )
    ranks_per_tile = total_ranks // 6
    tile = get_tile_index(rank, total_ranks) + 1
    count = rank % ranks_per_tile
    if total_ranks > 6:
        rank_suffix = f".tile{tile}.nc.{count:04}"
    else:
        rank_suffix = f".tile{tile}.nc"
    return rank_suffix


def _apply_dims(da, new_dims):
    """Applies new dimension names to the last dimensions of the given DataArray."""
    return da.rename(dict(zip(da.dims[-len(new_dims) :], new_dims)))


def _apply_restart_metadata(state, restart_properties: RestartProperties):
    new_state = {}
    for name, da in state.items():
        if name in restart_properties.keys():
            properties = restart_properties[name]
            new_dims = properties["dims"]
            new_state[name] = _apply_dims(da, new_dims)
            new_state[name].attrs["units"] = properties["units"]
        else:
            new_state[name] = copy.deepcopy(da)
    return new_state


def map_keys(old_dict, old_keys_to_new):
    new_dict = {}
    for old_key, new_key in old_keys_to_new.items():
        if old_key in old_dict:
            new_dict[new_key] = old_dict[old_key]
    for old_key in set(old_dict.keys()).difference(old_keys_to_new.keys()):
        new_dict[old_key] = old_dict[old_key]
    return new_dict


def prepend_label(filename, label=None):
    if label is not None and len(label) > 0:
        return f"{label}.{filename}"
    else:
        return filename


def load_partial_state_from_restart_file(
    file, restart_properties: RestartProperties, only_names=None
):
    ds = xr.open_dataset(file).isel(Time=0).drop_vars("Time")
    state = map_keys(ds.data_vars, _get_restart_standard_names(restart_properties))
    state = _apply_restart_metadata(state, restart_properties)
    if only_names is None:
        only_names = state.keys()
    state = {  # remove any variables that don't have restart metadata
        name: value
        for name, value in state.items()
        if ((name == "time") or ("units" in value.attrs)) and name in only_names
    }
    for name, array in state.items():
        if name != "time":
            array.load()
            state[name] = Quantity.from_data_array(array)
    return state


def _get_restart_standard_names(restart_properties: RestartProperties = None):
    """Return a list of variable names needed for a smooth restart. By default uses
    restart_properties from RESTART_PROPERTIES."""
    if restart_properties is None:
        restart_properties = RESTART_PROPERTIES
    return_dict = {}
    for std_name, properties in restart_properties.items():
        return_dict[properties["restart_name"]] = std_name
    return return_dict
