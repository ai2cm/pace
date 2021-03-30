from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.a2b_ord4 as a2b_ord4
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtstencil()
def set_k0(pp: FloatField, pk3: FloatField, top_value: float):
    with computation(PARALLEL), interval(...):
        pp[0, 0, 0] = 0.0
        pk3[0, 0, 0] = top_value


@gtstencil()
def CalcWk(pk: FloatField, wk: FloatField):
    with computation(PARALLEL), interval(...):
        wk = pk[0, 0, 1] - pk[0, 0, 0]


@gtstencil()
def CalcU(
    u: FloatField,
    du: FloatField,
    wk: FloatField,
    wk1: FloatField,
    gz: FloatField,
    pk3: FloatField,
    pp: FloatField,
    rdx: FloatFieldIJ,
    dt: float,
):
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
        ) * rdx


@gtstencil()
def CalcV(
    v: FloatField,
    dv: FloatField,
    wk: FloatField,
    wk1: FloatField,
    gz: FloatField,
    pk3: FloatField,
    pp: FloatField,
    rdy: FloatFieldIJ,
    dt: float,
):
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
        ) * rdy


def compute(u, v, pp, gz, pk3, delp, dt, ptop, akap):
    """
    u=u v=v pp=pkc gz=gz pk3=pk3 delp=delp dt=dt
    """
    grid = spec.grid
    orig = (grid.is_, grid.js, 0)
    # peln1 = log(ptop)
    ptk = ptop ** akap
    top_value = ptk  # = peln1 if spec.namelist.use_logp else ptk

    wk1 = utils.make_storage_from_shape(
        pp.shape, origin=orig, cache_key="nh_p_grad_wk1"
    )
    wk = utils.make_storage_from_shape(pk3.shape, origin=orig, cache_key="nh_p_grad_wk")

    set_k0(pp, pk3, top_value, origin=orig, domain=(grid.nic + 1, grid.njc + 1, 1))

    a2b_ord4.compute(pp, wk1, kstart=1, nk=grid.npz, replace=True)
    a2b_ord4.compute(pk3, wk1, kstart=1, nk=grid.npz, replace=True)

    a2b_ord4.compute(gz, wk1, kstart=0, nk=grid.npz + 1, replace=True)
    a2b_ord4.compute(delp, wk1)

    CalcWk(pk3, wk, origin=orig, domain=(grid.nic + 1, grid.njc + 1, grid.npz))

    du = utils.make_storage_from_shape(u.shape, origin=orig, cache_key="nh_p_grad_du")

    CalcU(
        u,
        du,
        wk,
        wk1,
        gz,
        pk3,
        pp,
        grid.rdx,
        dt,
        origin=orig,
        domain=(grid.nic, grid.njc + 1, grid.npz),
    )

    dv = utils.make_storage_from_shape(v.shape, origin=orig, cache_key="nh_p_grad_dv")

    CalcV(
        v,
        dv,
        wk,
        wk1,
        gz,
        pk3,
        pp,
        grid.rdy,
        dt,
        origin=orig,
        domain=(grid.nic + 1, grid.njc, grid.npz),
    )
    return u, v, pp, gz, pk3, delp
