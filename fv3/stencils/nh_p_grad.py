import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.a2b_ord4 as a2b_ord4
from math import log

sd = utils.sd


def grid():
    return spec.grid


@utils.stencil()
def set_k0(pp: sd, pk3: sd, top_value: float):
    with computation(PARALLEL), interval(...):
        pp[0, 0, 0] = 0.0
        pk3[0, 0, 0] = top_value


@utils.stencil()
def CalcWk(pk: sd, wk: sd):
    with computation(PARALLEL), interval(...):
        wk = pk[0, 0, 1] - pk[0, 0, 0]


@utils.stencil()
def CalcU(u: sd, du: sd, dt: float, wk: sd, wk1: sd, gz: sd, pk3: sd, pp: sd, rdx: sd):
    with computation(PARALLEL), interval(...):
        # hydrostatic contribution
        du = (
            dt
            / (wk[0, 0, 0] + wk[1, 0, 0])
            * (
                (gz[0, 0, 1] - gz[1, 0, 0]) * (pk3[1, 0, 1] - pk3[0, 0, 0])
                + (gz[0, 0, 0] - gz[1, 0, 1]) * (pk3[0, 0, 1] - pk3[1, 0, 0])
            )
        )
        # nonhydrostatic contribution
        u[0, 0, 0] = (
            u[0, 0, 0]
            + du[0, 0, 0]
            + dt
            / (wk1[0, 0, 0] + wk1[1, 0, 0])
            * (
                (gz[0, 0, 1] - gz[1, 0, 0]) * (pp[1, 0, 1] - pp[0, 0, 0])
                + (gz[0, 0, 0] - gz[1, 0, 1]) * (pp[0, 0, 1] - pp[1, 0, 0])
            )
        ) * rdx[0, 0, 0]


@utils.stencil()
def CalcV(v: sd, dv: sd, dt: float, wk: sd, wk1: sd, gz: sd, pk3: sd, pp: sd, rdy: sd):
    with computation(PARALLEL), interval(...):
        # hydrostatic contribution
        dv[0, 0, 0] = (
            dt
            / (wk[0, 0, 0] + wk[0, 1, 0])
            * (
                (gz[0, 0, 1] - gz[0, 1, 0]) * (pk3[0, 1, 1] - pk3[0, 0, 0])
                + (gz[0, 0, 0] - gz[0, 1, 1]) * (pk3[0, 0, 1] - pk3[0, 1, 0])
            )
        )
        # nonhydrostatic contribution
        v[0, 0, 0] = (
            v[0, 0, 0]
            + dv[0, 0, 0]
            + dt
            / (wk1[0, 0, 0] + wk1[0, 1, 0])
            * (
                (gz[0, 0, 1] - gz[0, 1, 0]) * (pp[0, 1, 1] - pp[0, 0, 0])
                + (gz[0, 0, 0] - gz[0, 1, 1]) * (pp[0, 0, 1] - pp[0, 1, 0])
            )
        ) * rdy[0, 0, 0]


def compute(u, v, pp, gz, pk3, delp, dt, ptop, akap):
    """
    u=u v=v pp=pkc gz=gz pk3=pk3 delp=delp dt=dt
    """
    grid = spec.grid
    orig = (grid.is_, grid.js, 0)

    # peln1 = log(ptop)
    ptk = ptop ** akap
    top_value = ptk  # = peln1 if spec.namelist["use_logp"] else ptk

    wk1 = utils.make_storage_from_shape(pp.shape, origin=orig)
    wk = utils.make_storage_from_shape(pk3.shape, origin=orig)
    # do j=js,je+1
    # do i=is,ie+1

    set_k0(
        pp, pk3, top_value, origin=orig, domain=(grid.nic + 1, grid.njc + 1, 1),
    )

    a2b_ord4.compute(pp, wk1, kstart=1, nk=grid.npz, replace=True)
    a2b_ord4.compute(pk3, wk1, kstart=1, nk=grid.npz, replace=True)
    assert pp[3, 3, 0] == 0.0
    assert pk3[3, 3, 0] == top_value
    a2b_ord4.compute(gz, wk1, kstart=0, nk=grid.npz + 1, replace=True)
    a2b_ord4.compute(delp, wk1)

    # do j=js,je+1
    # do i=is,ie+1
    CalcWk(pk3, wk, origin=orig, domain=(grid.nic + 1, grid.njc + 1, grid.npz))

    du = utils.make_storage_from_shape(u.shape, origin=orig)
    # do j=js,je+1
    # do i=is,ie
    CalcU(
        u,
        du,
        dt,
        wk,
        wk1,
        gz,
        pk3,
        pp,
        grid.rdx,
        origin=orig,
        domain=(grid.nic, grid.njc + 1, grid.npz),
    )

    dv = utils.make_storage_from_shape(v.shape, origin=orig)
    # do j=js,je
    # do i=is,ie+1
    CalcV(
        v,
        dv,
        dt,
        wk,
        wk1,
        gz,
        pk3,
        pp,
        grid.rdy,
        origin=orig,
        domain=(grid.nic + 1, grid.njc, grid.npz),
    )
    return u, v, pp, gz, pk3, delp
