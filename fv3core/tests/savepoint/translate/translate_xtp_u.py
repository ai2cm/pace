from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.xtp_u as xtp_u
from fv3core.utils.grid import GridData, axis_offsets
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ

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
        ax_offsets = axis_offsets(grid_indexing, origin, domain)
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
            skip_passes=("GreedyMerging",),
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
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["u"] = {}
        self.in_vars["data_vars"]["c"]["serialname"] = "ub"
        self.in_vars["data_vars"]["flux"]["serialname"] = "vb"

    def compute_from_storage(self, inputs):
        xtp_obj = XTP_U(
            stencil_factory=self.grid.stencil_factory,
            grid_data=self.grid.grid_data,
            grid_type=spec.namelist.grid_type,
            iord=spec.namelist.hord_mt,
        )
        xtp_obj(inputs["c"], inputs["u"], inputs["flux"])
        return inputs
