from gt4py.gtscript import PARALLEL, computation, interval

import pace.dsl
import pace.fv3core.stencils.xtp_u as xtp_u
import pace.util
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.util.grid import GridData

from .translate_ytp_v import TranslateYTP_V


def xtp_u_stencil_defn(
    ub_contra_times_dt: FloatField,
    u: FloatField,
    updated_u: FloatField,
    dx: FloatFieldIJ,
    dxa: FloatFieldIJ,
    rdx: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        updated_u = xtp_u.advect_u_along_x(u, ub_contra_times_dt, rdx, dx, dxa, 1.0)


class XTP_U:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        grid_type: int,
        iord: int,
    ):
        if iord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert grid_type < 3
        grid_indexing = stencil_factory.grid_indexing

        origin = grid_indexing.origin_compute()
        domain = grid_indexing.domain_compute(add=(1, 1, 0))
        self._dx = grid_data.dx
        self._dxa = grid_data.dxa
        self._rdx = grid_data.rdx
        ax_offsets = grid_indexing.axis_offsets(origin, domain)
        self._stencil = stencil_factory.from_origin_domain(
            xtp_u_stencil_defn,
            externals={
                "iord": iord,
                "mord": iord,
                "xt_minmax": False,
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )

    def __call__(self, c: FloatField, u: FloatField, flux: FloatField):
        """
        Compute flux of kinetic energy in x-dir.

        Args:
            c (in): product of x-dir wind on cell corners and timestep
            u (in): x-dir wind on D-grid
            flux (out): Flux of kinetic energy
        """
        self._stencil(
            c,
            u,
            flux,
            self._dx,
            self._dxa,
            self._rdx,
        )


class TranslateXTP_U(TranslateYTP_V):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"]["u"] = {}
        self.in_vars["data_vars"]["c"]["serialname"] = "ub"
        self.in_vars["data_vars"]["flux"]["serialname"] = "vb"
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    def compute_from_storage(self, inputs):
        xtp_obj = XTP_U(
            stencil_factory=self.stencil_factory,
            grid_data=self.grid.grid_data,
            grid_type=self.namelist.grid_type,
            iord=self.namelist.hord_mt,
        )
        xtp_obj(inputs["c"], inputs["u"], inputs["flux"])
        return inputs
