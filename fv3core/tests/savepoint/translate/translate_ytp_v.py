from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.ytp_v as ytp_v
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils.grid import GridData, axis_offsets
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ


def ytp_v_stencil_defn(
    vb_contra_times_dt: FloatField,
    v: FloatField,
    updated_v: FloatField,
    dy: FloatFieldIJ,
    dya: FloatFieldIJ,
    rdy: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        updated_v = ytp_v.advect_v_along_y(v, vb_contra_times_dt, rdy, dy, dya, 1.0)


class YTP_V:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        grid_type: int,
        jord: int,
    ):
        if jord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently ytp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert grid_type < 3
        grid_indexing = stencil_factory.grid_indexing

        origin = grid_indexing.origin_compute()
        domain = grid_indexing.domain_compute(add=(1, 1, 0))
        self._dy = grid_data.dy
        self._dya = grid_data.dya
        self._rdy = grid_data.rdy
        ax_offsets = axis_offsets(grid_indexing, origin, domain)

        self.stencil = stencil_factory.from_origin_domain(
            ytp_v_stencil_defn,
            externals={
                "jord": jord,
                "mord": jord,
                "yt_minmax": False,
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
            skip_passes=("GreedyMerging",),
        )

    def __call__(self, c: FloatField, v: FloatField, flux: FloatField):
        """
        Compute flux of kinetic energy in y-dir.

        Args:
        c (in): product of y-dir wind on cell corners and timestep
        v (in): y-dir wind on Arakawa D-grid
        flux (out): Flux of kinetic energy
        """

        self.stencil(c, v, flux, self._dy, self._dya, self._rdy)


class TranslateYTP_V(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        c_info = self.grid.compute_dict_buffer_2d()
        c_info["serialname"] = "vb"
        flux_info = self.grid.compute_dict_buffer_2d()
        flux_info["serialname"] = "ub"
        self.in_vars["data_vars"] = {"c": c_info, "v": {}, "flux": flux_info}
        self.in_vars["parameters"] = []
        self.out_vars = {"flux": flux_info}

    def compute_from_storage(self, inputs):
        ytp_obj = YTP_V(
            stencil_factory=self.grid.stencil_factory,
            grid_data=self.grid.grid_data,
            grid_type=spec.namelist.grid_type,
            jord=spec.namelist.hord_mt,
        )
        ytp_obj(inputs["c"], inputs["v"], inputs["flux"])
        return inputs
