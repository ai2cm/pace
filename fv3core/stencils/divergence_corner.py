import gt4py as gt
import gt4py.gtscript as gtscript
import numpy as np

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd
origin = utils.origin

##
## Stencil Definitions
##


@gtstencil()
def matrix_element_subtraction(A: sd, B: sd):
    with computation(PARALLEL), interval(...):
        A = A - B[0, -1, 0]


@gtstencil()
def compute_diverg(div: sd, rarea_c: sd, v: sd, u: sd):
    with computation(PARALLEL), interval(...):
        div = rarea_c * (v[0, -1, 0] - v + u[-1, 0, 0] - u)


@gtstencil()
def compute_uf(
    uf: sd, u: sd, va: sd, cos_sg4: sd, cos_sg2: sd, dyc: sd, sin_sg4: sd, sin_sg2: sd
):
    with computation(PARALLEL), interval(...):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1, 0] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1, 0] + sin_sg2)
        )


@gtstencil()
def compute_vf(
    vf: sd, v: sd, ua: sd, cos_sg3: sd, cos_sg1: sd, dxc: sd, sin_sg3: sd, sin_sg1: sd
):
    with computation(PARALLEL), interval(...):
        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0, 0] + sin_sg1)
        )


@gtstencil()
def compute_uf_edge_values(uf: sd, u: sd, dyc: sd, sin_sg4: sd, sin_sg2: sd):
    with computation(PARALLEL), interval(...):
        uf = u * dyc * 0.5 * (sin_sg4[0, -1, 0] + sin_sg2)


@gtstencil()
def compute_vf_edge_values(vf: sd, v: sd, dxc: sd, sin_sg3: sd, sin_sg1: sd):
    with computation(PARALLEL), interval(...):
        vf = v * dxc * 0.5 * (sin_sg3[-1, 0, 0] + sin_sg1)


@gtstencil()
def compute_diverg_d(div: sd, vf: sd, uf: sd):
    with computation(PARALLEL), interval(...):
        div = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf


def compute(u, v, ua, va, divg_d):
    grid = spec.grid
    co = grid.compute_origin()
    is2 = grid.is_
    ie1 = grid.ie + 1
    # Create storage objects for the temporary velocity arrays, uf and vf
    uf = utils.make_storage_from_shape(ua.shape, origin=(grid.is_ - 2, grid.js - 1, 0))
    vf = utils.make_storage_from_shape(va.shape, origin=(grid.is_ - 1, grid.js - 2, 0))

    # Compute values for the temporary velocity arrays
    if spec.namelist.grid_type == 4:
        basic.multiply_stencil(
            u,
            grid.dyc,
            uf,
            origin=(grid.is_ - 2, grid.js - 1, 0),
            domain=(grid.nic + 4, grid.njc + 3, grid.npz),
        )
        basic.multiply_stencil(
            v,
            grid.dxc,
            vf,
            origin=(grid.is_ - 1, grid.js - 2, 0),
            domain=(grid.nic + 3, grid.njc + 4, grid.npz),
        )
        raise Exception("unimplemented, untest option grid_type = 4")
    else:
        compute_uf(
            uf,
            u,
            va,
            grid.cos_sg4,
            grid.cos_sg2,
            grid.dyc,
            grid.sin_sg4,
            grid.sin_sg2,
            origin=(grid.is_ - 1, grid.js, 0),
            domain=(grid.nic + 2, grid.njc + 1, grid.npz),
        )
        edge_domain = (grid.nic + 2, 1, grid.npz)
        if grid.south_edge:
            compute_uf_edge_values(
                uf,
                u,
                grid.dyc,
                grid.sin_sg4,
                grid.sin_sg2,
                origin=(grid.is_ - 1, grid.js, 0),
                domain=edge_domain,
            )
        if grid.north_edge:
            compute_uf_edge_values(
                uf,
                u,
                grid.dyc,
                grid.sin_sg4,
                grid.sin_sg2,
                origin=(grid.is_ - 1, grid.je + 1, 0),
                domain=edge_domain,
            )

        compute_vf(
            vf,
            v,
            ua,
            grid.cos_sg3,
            grid.cos_sg1,
            grid.dxc,
            grid.sin_sg3,
            grid.sin_sg1,
            origin=(is2, grid.js - 1, 0),
            domain=(ie1 - is2 + 1, grid.njc + 2, grid.npz),
        )
        edge_domain = (1, grid.njc + 2, grid.npz)
        if grid.west_edge:
            compute_vf_edge_values(
                vf,
                v,
                grid.dxc,
                grid.sin_sg3,
                grid.sin_sg1,
                origin=(grid.is_, grid.js - 1, 0),
                domain=edge_domain,
            )
        if grid.east_edge:
            compute_vf_edge_values(
                vf,
                v,
                grid.dxc,
                grid.sin_sg3,
                grid.sin_sg1,
                origin=(grid.ie + 1, grid.js - 1, 0),
                domain=edge_domain,
            )

    # Compute the divergence tensor values

    compute_diverg_d(
        divg_d,
        vf,
        uf,
        origin=(grid.is_, grid.js, 0),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
    if grid.sw_corner:
        matrix_element_subtraction(
            divg_d, vf, origin=(grid.is_, grid.js, 0), domain=grid.corner_domain()
        )
    if grid.se_corner:
        matrix_element_subtraction(
            divg_d, vf, origin=(grid.ie + 1, grid.js, 0), domain=grid.corner_domain()
        )
    if grid.ne_corner:
        basic.add_term_stencil(
            vf,
            divg_d,
            origin=(grid.ie + 1, grid.je + 1, 0),
            domain=grid.corner_domain(),
        )
    if grid.nw_corner:
        basic.add_term_stencil(
            vf, divg_d, origin=(grid.is_, grid.je + 1, 0), domain=grid.corner_domain()
        )

    basic.adjustmentfactor_stencil(
        grid.rarea_c,
        divg_d,
        origin=(grid.is_, grid.js, 0),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
