import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


@gtstencil()
def update_vorticity_and_kinetic_energy(
    ke: sd,
    vort: sd,
    ua: sd,
    va: sd,
    uc: sd,
    vc: sd,
    u: sd,
    v: sd,
    sin_sg1: sd,
    cos_sg1: sd,
    sin_sg2: sd,
    cos_sg2: sd,
    sin_sg3: sd,
    cos_sg3: sd,
    sin_sg4: sd,
    cos_sg4: sd,
    dt2: float,
):
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        ke = uc if ua > 0.0 else uc[1, 0, 0]
        vort = vc if va > 0.0 else vc[0, 1, 0]

        if __INLINED(spec.namelist.grid_type < 3):
            # additional assumption: not __INLINED(spec.grid.nested)
            with parallel(region[:, j_start - 1], region[:, j_end]):
                vort = vort * sin_sg4 + u[0, 1, 0] * cos_sg4 if va <= 0.0 else vort
            with parallel(region[:, j_start], region[:, j_end + 1]):
                vort = vort * sin_sg2 + u * cos_sg2 if va > 0.0 else vort

            with parallel(region[i_end, :], region[i_start - 1, :]):
                ke = ke * sin_sg3 + v[1, 0, 0] * cos_sg3 if ua <= 0.0 else ke
            with parallel(region[i_end + 1, :], region[i_start, :]):
                ke = ke * sin_sg1 + v * cos_sg1 if ua > 0.0 else ke

        ke = 0.5 * dt2 * (ua * ke + va * vort)


def compute(uc: sd, vc: sd, u: sd, v: sd, ua: sd, va: sd, dt2: float):
    grid = spec.grid
    origin = (grid.is_ - 1, grid.js - 1, 0)

    # Create storage objects to hold the new vorticity and kinetic energy values
    ke_c = utils.make_storage_from_shape(uc.shape, origin=origin)
    vort_c = utils.make_storage_from_shape(vc.shape, origin=origin)

    # Set vorticity and kinetic energy values
    update_vorticity_and_kinetic_energy(
        ke_c,
        vort_c,
        ua,
        va,
        uc,
        vc,
        u,
        v,
        grid.sin_sg1,
        grid.cos_sg1,
        grid.sin_sg2,
        grid.cos_sg2,
        grid.sin_sg3,
        grid.cos_sg3,
        grid.sin_sg4,
        grid.cos_sg4,
        dt2,
        origin=(grid.is_ - 1, grid.js - 1, 0),
        domain=(grid.nic + 2, grid.njc + 2, grid.npz),
    )
    return ke_c, vort_c
