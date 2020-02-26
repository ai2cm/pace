import os
import xarray as xr
import copy
from . import fortran_info
from . import _domain as domain
from . import io, filesystem, constants


__all__ = ['open_restart']

RESTART_NAMES = ('fv_core.res', 'fv_srf_wnd.res', 'fv_tracer.res')
RESTART_OPTIONAL_NAMES = ('sfc_data', 'phy_data')  # not output for dycore-only runs
COUPLER_RES_NAME = 'coupler.res'


def open_restart(dirname, partitioner, comm, label='', only_names=None):
    """Load restart files output by the Fortran model into a state dictionary.

    Args:
        dirname: location of restart files, can be local or remote
        partitioner: domain decomposition for this rank
        comm: mpi4py comm object
        label: prepended string on the restart files to load
        only_names (optional): list of standard names to load

    Returns:
        state: model state dictionary
    """
    tile_index = partitioner.tile(comm.Get_rank())
    rank = comm.Get_rank()
    state = {}
    if rank == partitioner.tile_master_rank(rank):
        for file in restart_files(dirname, tile_index, label):
            state.update(load_partial_state_from_restart_file(file, only_names=only_names))
        coupler_res_filename = get_coupler_res_filename(dirname, label)
        if filesystem.is_file(coupler_res_filename):
            with filesystem.open(coupler_res_filename, 'r') as f:
                state['time'] = io.get_current_date_from_coupler_res(f)
    state = broadcast_state(state, partitioner, comm)
    return state


def get_coupler_res_filename(dirname, label):
    return os.path.join(dirname, prepend_label(COUPLER_RES_NAME, label))


def broadcast_state(state, partitioner, comm):
    tile_comm = comm.Split(color=partitioner.tile(comm.Get_rank()), key=comm.Get_rank())
    if tile_comm.Get_rank() == constants.MASTER_RANK:
        name_list = list(set(state.keys()).difference('time'))
        name_list = tile_comm.bcast(name_list, root=constants.MASTER_RANK)
        array_list = [state[name] for name in name_list]
        metadata_list = domain.bcast_metadata_list(tile_comm, array_list)
        for name, array, metadata in zip(name_list, array_list, metadata_list):
            state[name] = partitioner.scatter_tile(tile_comm, array, metadata)
        state['time'] = comm.bcast(state['time'], root=constants.MASTER_RANK)
    else:
        name_list = tile_comm.bcast(None, root=constants.MASTER_RANK)
        metadata_list = domain.bcast_metadata_list(tile_comm, None)
        for name, metadata in zip(name_list, metadata_list):
            state[name] = partitioner.scatter_tile(tile_comm, None, metadata)
        state['time'] = tile_comm.bcast(None, root=constants.MASTER_RANK)
    tile_comm.Free()


def restart_files(dirname, tile_index, label):
    for filename in restart_filenames(dirname, tile_index, label):
        with open(filename, 'rb') as f:
            yield f


def restart_filenames(dirname, tile_index, label):
    suffix = f'.tile{tile_index + 1}.nc'
    return_list = []
    for name in RESTART_NAMES + RESTART_OPTIONAL_NAMES:
        filename = os.path.join(dirname, prepend_label(name, label) + suffix)
        if (name in RESTART_NAMES) or filesystem.is_file(filename):
            yield filename


def get_rank_suffix(rank, total_ranks):
    if total_ranks % 6 != 0:
        raise ValueError(
            f'total_ranks must be evenly divisible by 6, was given {total_ranks}'
        )
    ranks_per_tile = total_ranks // 6
    tile = domain.get_tile_number(rank, total_ranks)
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


def load_partial_state_from_restart_file(file, only_names=None):
    ds = xr.open_dataset(file).isel(Time=0).drop("Time")
    state = map_keys(ds.data_vars, fortran_info.get_restart_standard_names())
    state = apply_restart_metadata(state)
    state = {  # remove any variables that don't have restart metadata
        name: value for name, value in state.items()
        if (name == 'time') or ('units' in value.attrs)
    }
    name_list = list(set(state.keys()).difference('time'))
    if only_names is not None:
        name_list = list(set(name_list).intersection(only_names))
    return state


def _restrict_to_rank(state, partitioner):
    return_dict = {}
    for name, array in state.items():
        if name == 'time':
            return_dict['time'] = array
        else:
            # discard tile dimension because one tile per file
            rank_slice = partitioner.subtile_range(array.dims, overlap=True)[1:]
            return_dict[name] = array[rank_slice]
    return return_dict
