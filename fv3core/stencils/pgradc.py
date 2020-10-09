import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd
origin = utils.origin


@gtscript.function
def p_grad_c_u(uc_in, wk, pkc, gz, rdxc, dt2):
    return uc_in + dt2 * rdxc / (wk[-1, 0, 0] + wk) * (
        (gz[-1, 0, 1] - gz) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
        + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc)
    )


@gtscript.function
def get_wk(pkc, delpc, hydrostatic):
    return pkc[0, 0, 1] - pkc if hydrostatic else delpc


@gtscript.function
def p_grad_c_u_wk(uc_in, delpc, pkc, gz, rdxc, hydrostatic, dt2):
    wk = get_wk(pkc, delpc, hydrostatic)
    return p_grad_c_u(uc_in, wk, pkc, gz, rdxc, dt2)


@gtscript.function
def p_grad_c_v(vc_in, wk, pkc, gz, rdyc, dt2):
    return vc_in + dt2 * rdyc / (wk[0, -1, 0] + wk) * (
        (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
        + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
    )


@gtscript.function
def p_grad_c_v_wk(vc_in, delpc, pkc, gz, rdyc, hydrostatic, dt2):
    wk = get_wk(pkc, delpc, hydrostatic)
    return p_grad_c_v(vc_in, wk, pkc, gz, rdyc, dt2)


@gtscript.function
def p_grad_c_fn(uc_in, vc_in, delpc, pkc, gz, rdxc, rdyc, hydrostatic, dt2):
    wk = get_wk(pkc, delpc, hydrostatic)
    uc_in = p_grad_c_u(uc_in, wk, pkc, gz, rdxc, dt2)
    vc_in = p_grad_c_v(vc_in, wk, pkc, gz, rdyc, dt2)
    return uc_in, vc_in


@gtstencil()
def p_grad_c(
    uc_in: sd,
    vc_in: sd,
    delpc: sd,
    pkc: sd,
    gz: sd,
    rdxc: sd,
    rdyc: sd,
    hydrostatic: int,
    dt2: float,
):
    with computation(PARALLEL), interval(0, -1):
        uc_in, vc_in = p_grad_c_fn(
            uc_in,
            vc_in,
            delpc,
            pkc,
            gz,
            rdxc,
            rdyc,
            hydrostatic,
            dt2,  # TODO: add [0, 0, 0] when gt4py bug is fixed
        )


@gtstencil()
def p_grad_c_ustencil(
    uc_in: sd, delpc: sd, pkc: sd, gz: sd, rdxc: sd, *, hydrostatic: int, dt2: float
):
    with computation(PARALLEL), interval(0, -1):
        uc_in = p_grad_c_u_wk(
            uc_in, delpc, pkc, gz, rdxc, hydrostatic, dt2
        )  # TODO: add [0, 0, 0] when gt4py bug is fixed


@gtstencil()
def p_grad_c_vstencil(
    vc_in: sd, delpc: sd, pkc: sd, gz: sd, rdyc: sd, hydrostatic: int, dt2: float
):
    with computation(PARALLEL), interval(0, -1):
        vc_in = p_grad_c_v_wk(
            vc_in, delpc, pkc, gz, rdyc, hydrostatic, dt2
        )  # TODO: add [0, 0, 0] when gt4py bug is fixed


def compute(uc, vc, delpc, pkc, gz, dt2):
    # Options:
    #      1) make a whole new storage for uc and vc out variables, paste computed values into proper indices of uc_in and vc_in
    #      2) copy the edges that aren't supposed to be computed for uc_in and vc_in, operate on vc_in and uv_in, then paste in edges
    #      3) compute uc and vc stencils separately specifying different domains to compute on
    grid = spec.grid
    co = grid.compute_origin()
    hydrostatic = int(spec.namelist.hydrostatic)
    # Option 2
    # uc_edge_i,  uc_edge_j, vc_edge_i, vc_edge_j = grid.edge_offset_halos(uc_in, vc_in)
    # p_grad_c(uc_in, vc_in, delpc, pkc, gz, grid.rdxc, grid.rdyc,
    #         hydrostatic=hydrostatic, dt2=dt2, origin=(grid.is_, grid.js, 0), domain=(grid.nic+1, grid.njc + 1, grid.npz + 1))
    # grid.append_edges(uc_in, uc_edge_i, uc_edge_j, vc_in, vc_edge_i, vc_edge_j)

    # Option 3
    p_grad_c_ustencil(
        uc,
        delpc,
        pkc,
        gz,
        grid.rdxc,
        hydrostatic=hydrostatic,
        dt2=dt2,
        origin=co,
        domain=(grid.nic + 1, grid.njc, grid.npz + 1),
    )
    p_grad_c_vstencil(
        vc,
        delpc,
        pkc,
        gz,
        grid.rdyc,
        hydrostatic=hydrostatic,
        dt2=dt2,
        origin=co,
        domain=(grid.nic, grid.njc + 1, grid.npz + 1),
    )
