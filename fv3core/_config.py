import os
from types import SimpleNamespace

import f90nml
import gt4py.gtscript as gtscript

import fv3core.utils.gt4py_utils as utils
from fv3core.utils.grid import Grid

from .utils import global_config


grid = None
namelist = SimpleNamespace()

# Global set of namelist defaults
namelist_defaults = {
    "grid_type": 0,
    "do_f3d": False,
    "inline_q": False,
    "do_skeb": False,  # save dissipation estimate
    "use_logp": False,
    "moist_phys": True,
    "check_negative": False,
    # gfdl_cloud_mucrophys.F90
    "tau_r2g": 900.0,  # rain freezing during fast_sat
    "tau_smlt": 900.0,  # snow melting
    "tau_g2r": 600.0,  # graupel melting to rain
    "tau_imlt": 600.0,  # cloud ice melting
    "tau_i2s": 1000.0,  # cloud ice to snow auto - conversion
    "tau_l2r": 900.0,  # cloud water to rain auto - conversion
    "tau_g2v": 900.0,  # graupel sublimation
    "tau_v2g": 21600.0,  # graupel deposition -- make it a slow process
    "sat_adj0": 0.90,  # adjustment factor (0: no, 1: full) during fast_sat_adj
    "ql_gen": 1.0e-3,  #  max cloud water generation during remapping step if fast_sat_adj = .t.
    "ql_mlt": 2.0e-3,  # max value of cloud water allowed from melted cloud ice
    "qs_mlt": 1.0e-6,  # max cloud water due to snow melt
    "ql0_max": 2.0e-3,  # max cloud water value (auto converted to rain)
    "t_sub": 184.0,  # min temp for sublimation of cloud ice
    "qi_gen": 1.82e-6,  # max cloud ice generation during remapping step
    "qi_lim": 1.0,  # cloud ice limiter to prevent large ice build up
    "qi0_max": 1.0e-4,  # max cloud ice value (by other sources)
    "rad_snow": True,  # consider snow in cloud fraciton calculation
    "rad_rain": True,  # consider rain in cloud fraction calculation
    "rad_graupel": True,  # consider graupel in cloud fraction calculation
    "tintqs": False,  # use temperature in the saturation mixing in PDF
    "dw_ocean": 0.10,  # base value for ocean
    "dw_land": 0.20,  # base value for subgrid deviation / variability over land
    "icloud_f": 0,  # cloud scheme 0 - ?, 1: old fvgfs gfdl) mp implementation, 2: binary cloud scheme (0 / 1)
    "cld_min": 0.05,  # !< minimum cloud fraction
    "tau_l2v": 300.0,  # cloud water to water vapor (evaporation)
    "tau_v2l": 150.0,  # water vapor to cloud water (condensation)
    "c2l_ord": 4,
    "regional": False,
    "m_split": 0,
    "convert_ke": False,
    "breed_vortex_inline": False,
    "use_old_omega": True,
    "use_logp": False,
    "RF_fast": False,
    "p_ref": 1e5,  # Surface pressure used to construct a horizontally-uniform reference
    "adiabatic": False,
    "nf_omega": 1,
    "fv_sg_adj": -1,
    "n_sponge": 1,
}


def namelist_to_flatish_dict(nml_input):
    nml = dict(nml_input)
    for name, value in nml.items():
        if isinstance(value, f90nml.Namelist):
            nml[name] = namelist_to_flatish_dict(value)
    flatter_namelist = {}
    for key, value in nml.items():
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


# TODO: Before this can be used, we need to write a module to make the grid data
# from files on disk and call it
def make_grid_from_namelist(namelist, rank):
    shape_params = {}
    for narg in ["npx", "npy", "npz"]:
        shape_params[narg] = getattr(namelist, narg)
    indices = {
        "isd": 0,
        "ied": namelist.npx + 2 * utils.halo - 2,
        "is_": utils.halo,
        "ie": namelist.npx + utils.halo - 2,
        "jsd": 0,
        "jed": namelist.npy + 2 * utils.halo - 2,
        "js": utils.halo,
        "je": namelist.npy + utils.halo - 2,
    }
    return Grid(indices, shape_params, rank, namelist.layout)


def set_grid(in_grid):
    """Updates the global grid given another.

    Args:
        in_grid (Grid): Input grid to set.
    """
    global grid
    grid = in_grid


def set_namelist(filename):
    """Updates the global namelist from a file and re-generates the global grid.

    Args:
        filename (str): Input file.
    """
    global grid

    namelist.__dict__.clear()
    namelist.__dict__.update(namelist_defaults)
    namelist.__dict__.update(namelist_to_flatish_dict(f90nml.read(filename).items()))

    grid = make_grid_from_namelist(namelist, 0)


if "NAMELIST_FILENAME" in os.environ:
    set_namelist(os.environ["NAMELIST_FILENAME"])
