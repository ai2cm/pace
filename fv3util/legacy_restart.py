import os
import xarray as xr
import copy
from . import fortran_info
from . import mpi, io, filesystem


RESTART_NAMES = ('fv_core.res', 'fv_srf_wnd.res', 'fv_tracer.res')
RESTART_OPTIONAL_NAMES = ('sfc_data', 'phy_data')
COUPLER_RES_NAME = 'coupler.res'


def get_rank_suffix(rank, total_ranks):
    if total_ranks % 6 != 0:
        raise ValueError(
            f'total_ranks must be evenly divisible by 6, was given {total_ranks}'
        )
    ranks_per_tile = total_ranks // 6
    tile = mpi.get_tile_number(rank, total_ranks)
    count = rank % ranks_per_tile
    if total_ranks > 6:
        rank_suffix = f'.tile{tile}.nc.{count:04}'
    else:
        rank_suffix = f'.tile{tile}.nc'
    return rank_suffix


def apply_dims(da, new_dims):
    """Applies new dimension names to the last dimensions of the given DataArray."""
    return da.rename(dict(zip(da.dims[-len(new_dims):], new_dims)))


def apply_restart_metadata(state):
    new_state = {}
    for name, da in state.items():
        if name in fortran_info.properties_by_std_name:
            properties = fortran_info.properties_by_std_name[name]
            new_dims = properties['dims']
            new_state[name] = apply_dims(da, new_dims)
            new_state[name].attrs['units'] = properties['units']
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
    if label is not None:
        return f'{label}.{filename}'
    else:
        return filename


def load_partial_state_from_restart_file(file):
    ds = xr.open_dataset(file).isel(Time=0).drop("Time")
    ds.load()
    state = map_keys(ds.data_vars, fortran_info.get_restart_standard_names())
    state = apply_restart_metadata(state)
    return state


def open_restart(dirname, rank, total_ranks, label=''):
    suffix = get_rank_suffix(rank, total_ranks)
    state = {}
    for name in RESTART_NAMES:
        filename = os.path.join(dirname, prepend_label(name, label) + suffix)
        with filesystem.open(filename, 'rb') as f:
            state.update(load_partial_state_from_restart_file(f))
    coupler_res_filename = os.path.join(dirname, prepend_label(COUPLER_RES_NAME, label))
    if filesystem.is_file(coupler_res_filename):
        with filesystem.open(coupler_res_filename, 'r') as f:
            state['time'] = io.get_current_date_from_coupler_res(f)
    return state
