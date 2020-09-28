import gt4py as gt
import gt4py.gtscript as gtscript
import numpy as np

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils


sd = utils.sd
origin = utils.origin

##
## Stencil Definitions
##

# Flux field computations
@utils.stencil()
def compute_tmp_flux(
    tmp_flux: sd, velocity: sd, velocity_c: sd, cosa: sd, sina: sd, dt2: float
):
    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina


@utils.stencil()
def compute_meridional_flux(flux: sd, tmp_flux: sd, vort: sd):
    with computation(PARALLEL), interval(...):
        flux = vort if tmp_flux > 0.0 else vort[0, 1, 0]


@utils.stencil()
def compute_zonal_flux(flux: sd, tmp_flux: sd, vort: sd):
    with computation(PARALLEL), interval(...):
        flux = vort if tmp_flux > 0.0 else vort[1, 0, 0]


@utils.stencil()
def compute_f1_edge_values(flux: sd, velocity: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        flux = dt2 * velocity


# Wind speed updates
@utils.stencil()
def update_uc(uc: sd, fy1: sd, fy: sd, rdxc: sd, ke: sd):
    with computation(PARALLEL), interval(...):
        uc = uc + fy1 * fy + rdxc * (ke[-1, 0, 0] - ke)


@utils.stencil()
def update_vc(vc: sd, fx1: sd, fx: sd, rdyc: sd, ke: sd):
    with computation(PARALLEL), interval(...):
        vc = vc - fx1 * fx + rdyc * (ke[0, -1, 0] - ke)


def compute(uc, vc, vort_c, ke_c, v, u, fxv, fyv, dt2):
    grid = spec.grid
    co = grid.compute_origin()
    zonal_domain = (grid.nic + 1, grid.njc, grid.npz)
    meridional_domain = (grid.nic, grid.njc + 1, grid.npz)

    # Create storage objects for the temporary flux fields
    fx1 = utils.make_storage_from_shape(uc.shape, origin=co)
    fy1 = utils.make_storage_from_shape(vc.shape, origin=co)

    # Compute the temporary fluxes in the zonal and meridional coordimate directions
    compute_tmp_flux(
        fx1, u, vc, grid.cosa_v, grid.sina_v, dt2, origin=co, domain=meridional_domain
    )
    compute_tmp_flux(
        fy1, v, uc, grid.cosa_u, grid.sina_u, dt2, origin=co, domain=zonal_domain
    )

    # Add edge effects if we are not in a regional or nested grid configuration
    if not grid.nested and spec.namelist.grid_type < 3:
        edge_domain = (1, grid.njc, grid.npz)
        if grid.west_edge:
            compute_f1_edge_values(fy1, v, dt2, origin=co, domain=edge_domain)
        if grid.east_edge:
            compute_f1_edge_values(
                fy1, v, dt2, origin=(grid.ie + 1, grid.js, 0), domain=edge_domain
            )

        edge_domain = (grid.nic, 1, grid.npz)
        if grid.south_edge:
            compute_f1_edge_values(fx1, u, dt2, origin=co, domain=edge_domain)
        if grid.north_edge:
            compute_f1_edge_values(
                fx1, u, dt2, origin=(grid.is_, grid.je + 1, 0), domain=edge_domain
            )

    # Compute main flux fields (ignoring edge values)
    compute_zonal_flux(fxv, fx1, vort_c, origin=co, domain=meridional_domain)
    compute_meridional_flux(fyv, fy1, vort_c, origin=co, domain=zonal_domain)

    # Update time-centered winds on C-grid
    update_uc(uc, fy1, fyv, grid.rdxc, ke_c, origin=co, domain=zonal_domain)
    update_vc(vc, fx1, fxv, grid.rdyc, ke_c, origin=co, domain=meridional_domain)
