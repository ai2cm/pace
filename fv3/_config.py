import f90nml
import os
import fv3.utils.gt4py_utils as utils
from fv3.utils.grid import Grid

grid = None
namelist = None


def namelist_to_flatish_dict(source):
    namelist = dict(source)
    for name, value in namelist.items():
        if isinstance(value, f90nml.Namelist):
            namelist[name] = namelist_to_flatish_dict(value)
    flatter_namelist = {}
    for key, value in namelist.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey in flatter_namelist:
                    raise Exception(
                        "Cannot flatten this namelist, duplicate keys: " + subkey
                    )
                flatter_namelist[subkey] = subvalue
        else:
            flatter_namelist[key] = value
    return flatter_namelist


def merge_namelist_defaults(nml):
    defaults = {
        "grid_type": 0,
        "do_f3d": False,
        "inline_q": False,
        "do_skeb": False,  # save dissipation estimate
        "use_logp": False,
        "c2l_ord": 4,
        "regional": False,
    }
    defaults.update(nml)
    return defaults


# TODO: Before this can be used, we need to write a module to make the grid data from files on disk and call it
def make_grid_from_namelist(namelist, rank):
    shape_params = {}
    for narg in ["npx", "npy", "npz"]:
        shape_params[narg] = namelist[narg]
    indices = {
        "isd": 0,
        "ied": namelist["npx"] + 2 * utils.halo - 2,
        "is_": utils.halo,
        "ie": namelist["npx"] + utils.halo - 2,
        "jsd": 0,
        "jed": namelist["npy"] + 2 * utils.halo - 2,
        "js": utils.halo,
        "je": namelist["npy"] + utils.halo - 2,
    }
    return Grid(indices, shape_params, rank, namelist["layout"])


def set_grid(in_grid):
    global grid
    grid = in_grid


def set_namelist(filename):
    global grid
    global namelist
    namelist = merge_namelist_defaults(
        namelist_to_flatish_dict(f90nml.read(filename).items())
    )
    grid = make_grid_from_namelist(namelist, 0)


if "NAMELIST_FILENAME" in os.environ:
    set_namelist(os.environ["NAMELIST_FILENAME"])
