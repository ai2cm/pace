from gt4py.gtscript import PARALLEL, computation, interval, stencil

import fv3core._config as spec
import fv3core.stencils.a2b_ord4 as a2b_ord4
import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.typing import FloatField, FloatFieldIJ


def set_k0(pp: FloatField, pk3: FloatField, top_value: float):
    with computation(PARALLEL), interval(...):
        pp[0, 0, 0] = 0.0
        pk3[0, 0, 0] = top_value


def calc_wk(pk: FloatField, wk: FloatField):
    with computation(PARALLEL), interval(...):
        wk = pk[0, 0, 1] - pk[0, 0, 0]


def calc_u(
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


def calc_v(
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


class NonHydrostaticPressureGradient:
    """
    Fortran name is nh_p_grad
    """

    def __init__(self):
        grid = spec.grid
        self.orig = grid.compute_origin()
        self.domain_full_k = grid.domain_shape_compute(add=(1, 1, 0))
        self.domain_k1 = (grid.nic + 1, grid.njc + 1, 1)
        self.u_domain = grid.domain_shape_compute(add=(0, 1, 0))
        self.v_domain = grid.domain_shape_compute(add=(1, 0, 0))
        self.nk = grid.npz
        self.rdx = grid.rdx
        self.rdy = grid.rdy

        self._tmp_wk = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 0, 1)), origin=self.orig
        )  # pk3.shape
        self._tmp_wk1 = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 0, 1)), origin=self.orig
        )  # pp.shape
        self._tmp_du = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 1, 0)), origin=self.orig
        )
        self._tmp_dv = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(1, 0, 0)), origin=self.orig
        )

        self._set_k0_stencil = stencil(
            definition=set_k0,
            backend=global_config.get_backend(),
            rebuild=global_config.get_rebuild(),
        )

        self._calc_wk_stencil = stencil(
            definition=calc_wk,
            backend=global_config.get_backend(),
            rebuild=global_config.get_rebuild(),
        )

        self._calc_u_stencil = stencil(
            definition=calc_u,
            backend=global_config.get_backend(),
            rebuild=global_config.get_rebuild(),
        )

        self._calc_v_stencil = stencil(
            definition=calc_v,
            backend=global_config.get_backend(),
            rebuild=global_config.get_rebuild(),
        )

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        pp: FloatField,
        gz: FloatField,
        pk3: FloatField,
        delp: FloatField,
        dt: float,
        ptop: float,
        akap: float,
    ):
        """
        Updates the U and V winds due to pressure gradients,
        accounting for both the hydrostatic and nonhydrostatic contributions.
        Args:
            u: U wind (inout)
            v: V wind (inout)
            pp: Pressure (in)
            gz:  height of the model grid cells (in)
            pk3: (in)
            delp: vertical delta in pressure (in)
            dt: model atmospheric timestep (in)
            ptop: pressure at top of atmosphere (in)
            akap: Kappa (in)
        Fortran names:
        u=u v=v pp=pkc gz=gz pk3=pk3 delp=delp dt=dt
        """
        ptk = ptop ** akap
        top_value = ptk  # = peln1 if spec.namelist.use_logp else ptk

        self._set_k0_stencil(
            pp, pk3, top_value, origin=self.orig, domain=self.domain_k1
        )

        a2b_ord4.compute(pp, self._tmp_wk1, kstart=1, nk=self.nk, replace=True)
        a2b_ord4.compute(pk3, self._tmp_wk1, kstart=1, nk=self.nk, replace=True)

        a2b_ord4.compute(gz, self._tmp_wk1, kstart=0, nk=self.nk + 1, replace=True)
        a2b_ord4.compute(delp, self._tmp_wk1)

        self._calc_wk_stencil(
            pk3, self._tmp_wk, origin=self.orig, domain=self.domain_full_k
        )

        self._calc_u_stencil(
            u,
            self._tmp_du,
            self._tmp_wk,
            self._tmp_wk1,
            gz,
            pk3,
            pp,
            self.rdx,
            dt,
            origin=self.orig,
            domain=self.u_domain,
        )

        self._calc_v_stencil(
            v,
            self._tmp_dv,
            self._tmp_wk,
            self._tmp_wk1,
            gz,
            pk3,
            pp,
            self.rdy,
            dt,
            origin=self.orig,
            domain=self.v_domain,
        )
        # return u, v, pp, gz, pk3, delp
