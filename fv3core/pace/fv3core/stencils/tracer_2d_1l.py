import math
from typing import Dict

from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import pace.dsl.gt4py_utils as utils
import pace.util
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.dace.wrapped_halo_exchange import WrappedHaloUpdater
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core.stencils.fvtp2d import FiniteVolumeTransport
from pace.util import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM


def get_area_flux(
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
    x_area_flux: FloatField,
    y_area_flux: FloatField,
):
    """
    Args:
        cx (in): Courant number in x-direction
        cy (in): Courant number in y-direction
        dxa (in):
        dya (in):
        dx (in):
        dy (in):
        sin_sg1 (in):
        sin_sg2 (in):
        sin_sg3 (in):
        sin_sg4 (in):
        x_area_flux (out): x-direction area flux
        y_area_flux (out): y-direction area flux
    """
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        with horizontal(region[local_is : local_ie + 2, local_js - 3 : local_je + 4]):
            x_area_flux = (
                cx * dxa[-1, 0] * dy * sin_sg3[-1, 0]
                if cx > 0
                else cx * dxa * dy * sin_sg1
            )
        with horizontal(region[local_is - 3 : local_ie + 4, local_js : local_je + 2]):
            y_area_flux = (
                cy * dya[0, -1] * dx * sin_sg4[0, -1]
                if cy > 0
                else cy * dya * dx * sin_sg2
            )


def divide_fluxes_by_n_substeps(
    cxd: FloatField,
    xfx: FloatField,
    x_mass_flux: FloatField,
    cyd: FloatField,
    yfx: FloatField,
    y_mass_flux: FloatField,
    n_split: int,
):
    """
    Divide all inputs in-place by the number of substeps n_split.

    Args:
        cxd (inout):
        xfx (inout):
        x_mass_flux (inout):
        cyd (inout):
        yfx (inout):
        y_mass_flux (inout):
    """
    with computation(PARALLEL), interval(...):
        frac = 1.0 / n_split
        cxd = cxd * frac
        xfx = xfx * frac
        x_mass_flux = x_mass_flux * frac
        cyd = cyd * frac
        yfx = yfx * frac
        y_mass_flux = y_mass_flux * frac


def cmax_stencil1(cx: FloatField, cy: FloatField, cmax: FloatField):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy))


def cmax_stencil2(
    cx: FloatField, cy: FloatField, sin_sg5: FloatField, cmax: FloatField
):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy)) + 1.0 - sin_sg5


def apply_mass_flux(
    dp_initial: FloatField,
    x_mass_flux: FloatField,
    y_mass_flux: FloatField,
    rarea: FloatFieldIJ,
    dp_final: FloatField,
):
    """
    Args:
        dp_initial (in): initial pressure thickness of layer (Pa)
        mfx (in):
        mfy (in):
        rarea (in): 1 / area
        dp_final (out): final pressure thickness of layer (Pa)
    """
    with computation(PARALLEL), interval(...):
        dp_final = (
            dp_initial
            + (x_mass_flux - x_mass_flux[1, 0, 0] + y_mass_flux - y_mass_flux[0, 1, 0])
            * rarea
        )


def apply_q_flux(
    q: FloatField,
    dp_initial: FloatField,
    q_x_flux: FloatField,
    q_y_flux: FloatField,
    rarea: FloatFieldIJ,
    dp_final: FloatField,
):
    """
    Args:
        q (inout): tracer
        dp_initial (in): initial pressure thickness of layer (Pa)
        q_x_flux (in): x-direction flux of q
        q_y_flux (in): y-direction flux of q
        rarea (in): 1 / area
        dp_final (in): final pressure thickness, after mass fluxes which occur
            simultaneously with q fluxes (Pa)
    """
    with computation(PARALLEL), interval(...):
        q = (
            q * dp_initial
            + (q_x_flux - q_x_flux[1, 0, 0] + q_y_flux - q_y_flux[0, 1, 0]) * rarea
        ) / dp_final


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
        quantity_factory: pace.util.QuantityFactory,
        transport: FiniteVolumeTransport,
        grid_data,
        comm: pace.util.CubedSphereCommunicator,
        tracers: Dict[str, pace.util.Quantity],
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )
        grid_indexing = stencil_factory.grid_indexing
        self.grid_indexing = grid_indexing  # needed for selective validation
        self._tracer_count = len(tracers)
        self.grid_data = grid_data

        self._x_area_flux = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_DIM, Z_DIM], units="unknown"
        )
        self._y_area_flux = quantity_factory.zeros(
            [X_DIM, Y_INTERFACE_DIM, Z_DIM], units="unknown"
        )
        self._q_x_flux = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM], units="unknown"
        )
        self._q_y_flux = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM], units="unknown"
        )
        self._tmp_dp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="Pa")
        self._tmp_dp2 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="Pa")

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
            get_area_flux,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._divide_fluxes_by_n_substeps = stencil_factory.from_origin_domain(
            divide_fluxes_by_n_substeps,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._apply_mass_flux = stencil_factory.from_origin_domain(
            apply_mass_flux,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )
        self._apply_q_flux = stencil_factory.from_origin_domain(
            apply_q_flux,
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

    def __call__(
        self,
        tracers: Dict[str, pace.util.Quantity],
        dp1,
        x_mass_flux,
        y_mass_flux,
        cxd,
        cyd,
    ):
        """
        Args:
            tracers (inout): tracers to advect according to fluxes during
                acoustic substeps
            dp1 (in): pressure thickness of atmospheric layers before acoustic substeps
            x_mass_flux (inout): total mass flux in x-direction over acoustic substeps
            y_mass_flux (inout): total mass flux in y-direction over acoustic substeps
            cxd (inout): accumulated courant number in x-direction
            cyd (inout): accumulated courant number in y-direction
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
            self._x_area_flux,
            self._y_area_flux,
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
            self._divide_fluxes_by_n_substeps(
                cxd,
                self._x_area_flux,
                x_mass_flux,
                cyd,
                self._y_area_flux,
                y_mass_flux,
                n_split,
            )

        self._tracers_halo_updater.update()

        dp2 = self._tmp_dp

        for it in range(n_split):
            last_call = it == n_split - 1
            self._apply_mass_flux(
                dp1,
                x_mass_flux,
                y_mass_flux,
                self.grid_data.rarea,
                dp2,
            )
            for q in tracers.values():
                self.finite_volume_transport(
                    q,
                    cxd,
                    cyd,
                    self._x_area_flux,
                    self._y_area_flux,
                    self._q_x_flux,
                    self._q_y_flux,
                    x_mass_flux=x_mass_flux,
                    y_mass_flux=y_mass_flux,
                )
                # TODO: rename to something about applying fluxes
                self._apply_q_flux(
                    q,
                    dp1,
                    self._q_x_flux,
                    self._q_y_flux,
                    self.grid_data.rarea,
                    dp2,
                )
            if not last_call:
                self._tracers_halo_updater.update()
                # we can't use variable assignment to avoid a data copy
                # because of current dace limitations
                self._swap_dp(dp1, dp2)
