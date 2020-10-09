import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd
origin = utils.origin

# Flux field computation
@gtstencil()
def compute_flux(flux: sd, grid_spacing: sd, val_in: sd):
    with computation(PARALLEL), interval(...):
        flux[0, 0, 0] = val_in * grid_spacing


# Vorticity field update (no corner value updates)
@gtstencil()
def update_vorticity(vorticity: sd, fx: sd, fy: sd):
    with computation(PARALLEL), interval(...):
        vorticity[0, 0, 0] = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy


# Vorticity field update of corner values
@gtstencil()
def update_vorticity_western_corner(vorticity: sd, fy: sd):
    with computation(PARALLEL), interval(...):
        vorticity[0, 0, 0] = vorticity + fy[-1, 0, 0]


@gtstencil()
def update_vorticity_eastern_corner(vorticity: sd, fy: sd):
    with computation(PARALLEL), interval(...):
        vorticity[0, 0, 0] = vorticity - fy


def compute(uc, vc, vort_c):
    grid = spec.grid

    # Compute the zonal flux values
    fx = utils.make_storage_from_shape(uc.shape, origin=(grid.is_ - 1, grid.js - 1, 0))
    compute_flux(
        fx,
        uc,
        grid.dxc,
        origin=(grid.is_, grid.js - 1, 0),
        domain=(grid.nic + 1, grid.njc + 2, grid.npz),
    )

    # Compute the meridional flux values
    fy = utils.make_storage_from_shape(vc.shape, origin=(grid.is_ - 1, grid.js - 1, 0))
    compute_flux(
        fy,
        vc,
        grid.dyc,
        origin=(grid.is_ - 1, grid.js, 0),
        domain=(grid.nic + 2, grid.njc + 1, grid.npz),
    )

    # Update the vorticity (non-corner) values
    update_vorticity(
        vort_c,
        fx,
        fy,
        origin=(grid.is_, grid.js, 0),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )

    # Update the vorticity corner values
    if grid.sw_corner:
        update_vorticity_western_corner(
            vort_c, fy, origin=(grid.is_, grid.js, 0), domain=grid.corner_domain()
        )
    if grid.nw_corner:
        update_vorticity_western_corner(
            vort_c, fy, origin=(grid.is_, grid.je + 1, 0), domain=grid.corner_domain()
        )
    if grid.se_corner:
        update_vorticity_eastern_corner(
            vort_c, fy, origin=(grid.ie + 1, grid.js, 0), domain=grid.corner_domain()
        )
    if grid.ne_corner:
        update_vorticity_eastern_corner(
            vort_c,
            fy,
            origin=(grid.ie + 1, grid.je + 1, 0),
            domain=grid.corner_domain(),
        )
