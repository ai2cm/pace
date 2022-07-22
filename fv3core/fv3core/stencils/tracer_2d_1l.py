import math
from typing import Dict

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import pace.dsl.gt4py_utils as utils
import pace.util
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.dace.wrapped_halo_exchange import WrappedHaloUpdater
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.util import Quantity


@gtscript.function
def flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is : local_ie + 2, local_js - 3 : local_je + 4]):
        xfx = (
            cx * dxa[-1, 0] * dy * sin_sg3[-1, 0] if cx > 0 else cx * dxa * dy * sin_sg1
        )
    return xfx


@gtscript.function
def flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is - 3 : local_ie + 4, local_js : local_je + 2]):
        yfx = (
            cy * dya[0, -1] * dx * sin_sg4[0, -1] if cy > 0 else cy * dya * dx * sin_sg2
        )
    return yfx


def flux_compute(
    cx: FloatField,
    cy: FloatField,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    xfx: FloatField,
    yfx: FloatField,
):
    """
    Args:
        cx (in):
        cy (in):
        dxa (in):
        dya (in):
        dx (in):
        dy (in):
        sin_sg1 (in):
        sin_sg2 (in):
        sin_sg3 (in):
        sin_sg4 (in):
        xfx (out):
        yfx (out):
    """
    with computation(PARALLEL), interval(...):
        xfx = flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx)
        yfx = flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx)


def cmax_multiply_by_frac(
    cxd: FloatField,
    xfx: FloatField,
    mfxd: FloatField,
    cyd: FloatField,
    yfx: FloatField,
    mfyd: FloatField,
    n_split: int,
):
    """
    Multiply all inputs in-place by 1.0 / n_split.

    Args:
        cxd (inout):
        xfx (inout):
        mfxd (inout):
        cyd (inout):
        yfx (inout):
        mfyd (inout):
    """
    with computation(PARALLEL), interval(...):
        frac = 1.0 / n_split
        cxd = cxd * frac
        xfx = xfx * frac
        mfxd = mfxd * frac
        cyd = cyd * frac
        yfx = yfx * frac
        mfyd = mfyd * frac


def cmax_stencil1(cx: FloatField, cy: FloatField, cmax: FloatField):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy))


def cmax_stencil2(
    cx: FloatField, cy: FloatField, sin_sg5: FloatField, cmax: FloatField
):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy)) + 1.0 - sin_sg5


def dp_fluxadjustment(
    dp1: FloatField,
    mfx: FloatField,
    mfy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    """
    Args:
        dp1 (in):
        mfx (in):
        mfy (in):
        rarea (in):
        dp2 (out):
    """
    with computation(PARALLEL), interval(...):
        dp2 = dp1 + (mfx - mfx[1, 0, 0] + mfy - mfy[0, 1, 0]) * rarea


@gtscript.function
def adjustment(q, dp1, fx, fy, rarea, dp2):
    return (q * dp1 + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / dp2


def q_adjust(
    q: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    """
    Args:
        q (inout):
        dp1 (in):
        fx (in):
        fy (in):
        rarea (in):
        dp2 (in):
    """
    with computation(PARALLEL), interval(...):
        q = adjustment(q, dp1, fx, fy, rarea, dp2)


# Simple stencil replacing:
#   self._tmp_dp2[:] = dp1
#   dp1[:] = dp2
#   dp2[:] = self._tmp_dp2
# Because dpX can be a quantity or an array
def swap_dp(dp1: FloatField, dp2: FloatField):
    with computation(PARALLEL), interval(...):
        tmp = dp1
        dp1 = dp2
        dp2 = tmp


class TracerAdvection:
    """
    Performs horizontal advection on tracers.

    Corresponds to tracer_2D_1L in the Fortran code.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        transport: FiniteVolumeTransport,
        grid_data,
        comm: pace.util.CubedSphereCommunicator,
        tracers: Dict[str, Quantity],
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_constant_args=["tracers"],
        )
        grid_indexing = stencil_factory.grid_indexing
        self.grid_indexing = grid_indexing  # needed for selective validation
        self._tracer_count = len(tracers)
        self.grid_data = grid_data
        shape = grid_indexing.domain_full(add=(1, 1, 1))
        origin = grid_indexing.origin_compute()

        def make_storage():
            return utils.make_storage_from_shape(
                shape=shape, origin=origin, backend=stencil_factory.backend
            )

        self._tmp_xfx = make_storage()
        self._tmp_yfx = make_storage()
        self._tmp_fx = make_storage()
        self._tmp_fy = make_storage()
        self._tmp_dp = make_storage()
        self._tmp_dp2 = make_storage()
        dims = [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM]
        origin, extent = grid_indexing.get_origin_domain(dims)
        self._tmp_qn2 = pace.util.Quantity(
            make_storage(),
            dims=dims,
            units="kg/m^2",
            origin=origin,
            extent=extent,
        )

        ax_offsets = grid_indexing.axis_offsets(
            grid_indexing.origin_full(), grid_indexing.domain_full()
        )
        local_axis_offsets = {}
        for axis_offset_name, axis_offset_value in ax_offsets.items():
            if "local" in axis_offset_name:
                local_axis_offsets[axis_offset_name] = axis_offset_value

        self._swap_dp = stencil_factory.from_origin_domain(
            swap_dp,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )

        self._flux_compute = stencil_factory.from_origin_domain(
            flux_compute,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._cmax_multiply_by_frac = stencil_factory.from_origin_domain(
            cmax_multiply_by_frac,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._dp_fluxadjustment = stencil_factory.from_origin_domain(
            dp_fluxadjustment,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )
        self._q_adjust = stencil_factory.from_origin_domain(
            q_adjust,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )
        self.finite_volume_transport: FiniteVolumeTransport = transport
        # If use AllReduce, will need something like this:
        # self._tmp_cmax = utils.make_storage_from_shape(shape, origin)
        # self._cmax_1 = stencil_factory.from_origin_domain(cmax_stencil1)
        # self._cmax_2 = stencil_factory.from_origin_domain(cmax_stencil2)

        # Setup halo updater for tracers
        tracer_halo_spec = grid_indexing.get_quantity_halo_spec(
            grid_indexing.domain_full(add=(1, 1, 1)),
            grid_indexing.origin_compute(),
            dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            n_halo=utils.halo,
            backend=stencil_factory.backend,
        )
        self._tracers_halo_updater = WrappedHaloUpdater(
            comm.get_scalar_halo_updater([tracer_halo_spec] * self._tracer_count),
            tracers,
            [t for t in tracers.keys()],
        )

    def __call__(self, tracers: Dict[str, Quantity], dp1, mfxd, mfyd, cxd, cyd, mdt):
        """
        Args:
            tracers (inout):
            dp1 (in):
            mfxd (inout):
            mfyd (inout):
            cxd (inout):
            cyd (inout):
        """
        # TODO: remove unused mdt argument
        # DaCe parsing issue
        # if len(tracers) != self._tracer_count:
        #     raise ValueError(
        #         f"incorrect number of tracers, {self._tracer_count} was "
        #         f"specified on init but {len(tracers)} were passed"
        #     )
        # start HALO update on q (in dyn_core in fortran -- just has started when
        # this function is called...)
        self._flux_compute(
            cxd,
            cyd,
            self.grid_data.dxa,
            self.grid_data.dya,
            self.grid_data.dx,
            self.grid_data.dy,
            self.grid_data.sin_sg1,
            self.grid_data.sin_sg2,
            self.grid_data.sin_sg3,
            self.grid_data.sin_sg4,
            self._tmp_xfx,
            self._tmp_yfx,
        )

        # # TODO for if we end up using the Allreduce and compute cmax globally
        # (or locally). For now, hardcoded.
        # split = int(grid_indexing.domain[2] / 6)
        # self._cmax_1(
        #     cxd, cyd, self._tmp_cmax, origin=grid_indexing.origin_compute(),
        #     domain=(grid_indexing.domain[0], self.grid_indexing.domain[1], split)
        # )
        # self._cmax_2(
        #     cxd,
        #     cyd,
        #     self.grid.sin_sg5,
        #     self._tmp_cmax,
        #     origin=(grid_indexing.isc, self.grid_indexing.jsc, split),
        #     domain=(
        #         grid_indexing.domain[0],
        #         self.grid_indexing.domain[1],
        #         grid_indexing.domain[2] - split + 1
        #     ),
        # )
        # cmax_flat = np.amax(self._tmp_cmax, axis=(0, 1))
        # # cmax_flat is a gt4py storage still, but of dimension [npz+1]...

        # cmax_max_all_ranks = cmax_flat.data
        # # TODO mpi allreduce...
        # # comm.Allreduce(cmax_flat, cmax_max_all_ranks, op=MPI.MAX)

        cmax_max_all_ranks = 2.0
        n_split = math.floor(1.0 + cmax_max_all_ranks)
        # NOTE: cmax is not usually a single value, it varies with k, if return to
        # that, make n_split a column as well

        if n_split > 1.0:
            self._cmax_multiply_by_frac(
                cxd,
                self._tmp_xfx,
                mfxd,
                cyd,
                self._tmp_yfx,
                mfyd,
                n_split,
            )

        self._tracers_halo_updater.update()

        dp2 = self._tmp_dp

        for it in range(n_split):
            last_call = it == n_split - 1
            self._dp_fluxadjustment(
                dp1,
                mfxd,
                mfyd,
                self.grid_data.rarea,
                dp2,
            )
            for q in tracers.values():
                self.finite_volume_transport(
                    q,
                    cxd,
                    cyd,
                    self._tmp_xfx,
                    self._tmp_yfx,
                    self._tmp_fx,
                    self._tmp_fy,
                    x_mass_flux=mfxd,
                    y_mass_flux=mfyd,
                )
                self._q_adjust(
                    q,
                    dp1,
                    self._tmp_fx,
                    self._tmp_fy,
                    self.grid_data.rarea,
                    dp2,
                )
            if not last_call:
                self._tracers_halo_updater.update()
                # use variable assignment to avoid a data copy
                self._swap_dp(dp1, dp2)
