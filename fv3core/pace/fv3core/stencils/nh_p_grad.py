from gt4py.cartesian.gtscript import PARALLEL, computation, interval

import pace.util
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core.stencils.a2b_ord4 import AGrid2BGridFourthOrder
from pace.util import X_DIM, Y_DIM, Z_INTERFACE_DIM
from pace.util.grid import GridData


def set_k0_and_calc_wk(
    pp: FloatField, pk3: FloatField, wk: FloatField, top_value: float
):
    """
    Args:
        pp (inout):
        pk3 (inout):
        wk (out): delta (pressure to the kappa) computed on corners
    """
    with computation(PARALLEL):
        with interval(0, 1):
            pp[0, 0, 0] = 0.0
            pk3[0, 0, 0] = top_value
            wk = pk3[0, 0, 1] - pk3[0, 0, 0]
        with interval(1, None):
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
    """
    Args:
        u (inout):
        wk (in):
        wk1 (in):
        gz (in):
        pk3 (in):
        pp (in):
        rdx (in):
    """
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
        # hydrostatic + nonhydrostatic contribution
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
    """
    Args:
        v (inout):
        wk (in):
        wk1 (in):
        gz (in):
        pk3 (in):
        pp (in):
        rdy (in):
    """
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
        # hydrostatic + nonhydrostatic contribution
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
    Apply nonhydrostatic pressure gradient force in the horizontal.

    Fully implicit with beta = 0.

    Documented in Lin 97, Section 6.6 of FV3 documentation.

    Fortran name is nh_p_grad.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        grid_data: GridData,
        grid_type,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self.orig = grid_indexing.origin_compute()
        domain_full_k = grid_indexing.domain_compute(add=(1, 1, 0))
        u_domain = grid_indexing.domain_compute(add=(0, 1, 0))
        v_domain = grid_indexing.domain_compute(add=(1, 0, 0))
        self.nk = grid_indexing.domain[2]
        self._rdx = grid_data.rdx
        self._rdy = grid_data.rdy

        self._tmp_wk = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM], units="unknown"
        )
        self._tmp_wk1 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM], units="unknown"
        )

        self.a2b_k1 = AGrid2BGridFourthOrder(
            stencil_factory.restrict_vertical(k_start=1),
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            grid_type=grid_type,
            z_dim=Z_INTERFACE_DIM,
            replace=True,
        )
        self.a2b_kbuffer = AGrid2BGridFourthOrder(
            stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            grid_type=grid_type,
            z_dim=Z_INTERFACE_DIM,
            replace=True,
        )
        self.a2b_kstandard = AGrid2BGridFourthOrder(
            stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            grid_type=grid_type,
            replace=False,
        )
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
            u (inout): wind in x-direction
            v (inout): wind in y-direction
            pp (inout): pressure, gets updated to B-grid
            gz (inout): height of the model grid cells, gets updated to B-grid
            pk3 (inout): gets updated to B-grid
            delp (in): vertical delta in pressure
            dt (in): model atmospheric timestep
            ptop (in): pressure at top of atmosphere
            akap (in): Kappa
        """
        # Fortran names:
        # u=u v=v pp=pkc gz=gz pk3=pk3 delp=delp dt=dt

        ptk = ptop ** akap
        top_value = ptk  # = peln1 if spec.namelist.use_logp else ptk

        # TODO: make it clearer that each of these a2b outputs is updated
        # instead of the output being put in tmp_wk1, possibly by removing
        # the second argument and using a temporary instead?

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
