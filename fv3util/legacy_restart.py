import os
import xarray as xr
from datetime import datetime
from . import _fortran_info as info


RESTART_NAMES = ('fv_core.res', 'fv_srf_wnd.res', 'fv_tracer.res')
RESTART_OPTIONAL_NAMES = ('sfc_data', 'phy_data')
COUPLER_RES_NAME = 'coupler.res'


def get_rank_suffix(tile, rank, total_ranks):
    count = rank % total_ranks
    if total_ranks > 6:
        rank_suffix = f'.tile{tile}.nc.{count:04}'
    else:
        rank_suffix = f'.tile{tile}.nc'
    return rank_suffix


def apply_restart_metadata():
    pass


def get_restart_standard_names():
    pass


def map_keys(old_keys_to_new, old_dict):
    new_dict = {}
    for old_key, new_key in old_keys_to_new.items():
        new_dict[new_key] = old_dict[old_key]
    for old_key in set(old_dict.keys()).difference(old_keys_to_new.keys()):
        new_dict[old_key] = old_dict[old_key]
    return new_dict


def prepend_label(filename, label=None):
    if label is not None:
        return f'{label}.{filename}'
    else:
        return filename


def get_integer_tokens(line, n_tokens):
    all_tokens = line.split()
    return [int(token) for token in all_tokens[:n_tokens]]


def get_current_date_from_coupler_res(filename):
    with open(filename, 'r') as f:
        f.readline()
        f.readline()
        year, month, day, hour, minute, second = get_integer_tokens(f.readline(), 6)
        return datetime(year, month, day, hour, minute, second)


def open_restart(dirname, tile, rank, total_ranks, label=''):
    suffix = get_rank_suffix(tile, rank, total_ranks)
    state = {}
    for name in RESTART_NAMES:
        filename = os.path.join(dirname, prepend_label(name, label) + suffix)
        state.update(_load_state(filename))
    coupler_res_filename = os.path.join(dirname, prepend_label(COUPLER_RES_NAME, label))
    state['time'] = get_current_date_from_coupler_res(coupler_res_filename)


def _load_state(filename):
    ds = xr.open_dataset(filename)
    ds = apply_restart_metadata(ds)
    return map_keys(get_restart_standard_names(), ds)

