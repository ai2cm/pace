from gt4py.gtscript import PARALLEL, computation, interval

import pace.dsl.gt4py_utils as utils
from fv3core.stencils.a2b_ord4 import AGrid2BGridFourthOrder
from fv3core.utils.grid import GridData
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.util import Z_INTERFACE_DIM


def set_k0_and_calc_wk(
    pp: FloatField, pk3: FloatField, wk: FloatField, top_value: float
):
    with computation(PARALLEL), interval(0, 1):
        pp[0, 0, 0] = 0.0
        pk3[0, 0, 0] = top_value
    with computation(PARALLEL), interval(...):
        wk = pk3[0, 0, 1] - pk3[0, 0, 0]


def calc_u(
    u: FloatField,
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
        dv = (
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

    def __init__(self, stencil_factory: StencilFactory, grid_data: GridData, grid_type):
        grid_indexing = stencil_factory.grid_indexing
        self.orig = grid_indexing.origin_compute()
        domain_full_k = grid_indexing.domain_compute(add=(1, 1, 0))
        u_domain = grid_indexing.domain_compute(add=(0, 1, 0))
        v_domain = grid_indexing.domain_compute(add=(1, 0, 0))
        self.nk = grid_indexing.domain[2]
        self._rdx = grid_data.rdx
        self._rdy = grid_data.rdy

        self._tmp_wk = utils.make_storage_from_shape(
            grid_indexing.domain_full(add=(0, 0, 1)),
            origin=self.orig,
            backend=stencil_factory.backend,
        )  # pk3.shape
        self._tmp_wk1 = utils.make_storage_from_shape(
            grid_indexing.domain_full(add=(0, 0, 1)),
            origin=self.orig,
            backend=stencil_factory.backend,
        )  # pp.shape

        self._set_k0_and_calc_wk_stencil = stencil_factory.from_origin_domain(
            set_k0_and_calc_wk,
            origin=self.orig,
            domain=domain_full_k,
        )

        self._calc_u_stencil = stencil_factory.from_origin_domain(
            calc_u,
            origin=self.orig,
            domain=u_domain,
        )

        self._calc_v_stencil = stencil_factory.from_origin_domain(
            calc_v,
            origin=self.orig,
            domain=v_domain,
        )
        self.a2b_k1 = AGrid2BGridFourthOrder(
            stencil_factory.restrict_vertical(k_start=1),
            grid_data,
            grid_type,
            z_dim=Z_INTERFACE_DIM,
            replace=True,
        )
        self.a2b_kbuffer = AGrid2BGridFourthOrder(
            stencil_factory,
            grid_data,
            grid_type,
            z_dim=Z_INTERFACE_DIM,
            replace=True,
        )
        self.a2b_kstandard = AGrid2BGridFourthOrder(
            stencil_factory,
            grid_data,
            grid_type,
            replace=False,
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

        self.a2b_k1(pp, self._tmp_wk1)
        self.a2b_k1(pk3, self._tmp_wk1)

        self.a2b_kbuffer(gz, self._tmp_wk1)
        self.a2b_kstandard(delp, self._tmp_wk1)

        self._set_k0_and_calc_wk_stencil(pp, pk3, self._tmp_wk, top_value)

        self._calc_u_stencil(
            u,
            self._tmp_wk,
            self._tmp_wk1,
            gz,
            pk3,
            pp,
            self._rdx,
            dt,
        )

        self._calc_v_stencil(
            v,
            self._tmp_wk,
            self._tmp_wk1,
            gz,
            pk3,
            pp,
            self._rdy,
            dt,
        )
