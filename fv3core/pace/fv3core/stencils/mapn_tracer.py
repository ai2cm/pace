from typing import Dict

import pace.dsl.gt4py_utils as utils
import pace.util
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField
from pace.fv3core.stencils.fillz import FillNegativeTracerValues
from pace.fv3core.stencils.map_single import MapSingle
from pace.util import X_DIM, Y_DIM, Z_DIM


class MapNTracer:
    """
    Fortran code is mapn_tracer, test class is MapN_Tracer_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        kord: int,
        nq: int,
        i1: int,
        i2: int,
        j1: int,
        j2: int,
        fill: bool,
        tracers: Dict[str, pace.util.Quantity],
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )
        grid_indexing = stencil_factory.grid_indexing
        self._origin = (i1, j1, 0)
        self._domain = ()
        self._nk = grid_indexing.domain[2]
        self._nq = int(nq)
        self._i1 = int(i1)
        self._i2 = int(i2)
        self._j1 = int(j1)
        self._j2 = int(j2)
        self._qs = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="unknown")

        kord_tracer = [kord] * self._nq
        kord_tracer[5] = 9  # qcld

        self._list_of_remap_objects = [
            MapSingle(
                stencil_factory, quantity_factory, kord_tracer[i], 0, i1, i2, j1, j2
            )
            for i in range(len(kord_tracer))
        ]

        if fill:
            self._fill_negative_tracers = True
            self._fillz = FillNegativeTracerValues(
                stencil_factory,
                quantity_factory,
                self._list_of_remap_objects[0].i_extent,
                self._list_of_remap_objects[0].j_extent,
                self._nk,
                self._nq,
                tracers,
            )
        else:
            self._fill_negative_tracers = False

    def __call__(
        self,
        pe1: FloatField,
        pe2: FloatField,
        dp2: FloatField,
        tracers: Dict[str, pace.util.Quantity],
    ):
        """
        Remaps the tracer species onto the Eulerian grid
        and optionally fills negative values in the tracer fields
        Assumes the minimum value is 0 for each tracer

        Args:
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            dp2 (in): Difference in pressure between Eulerian levels
            tracers (inout): tracers to be remapped
        """
        for i, q in enumerate(utils.tracer_variables[0 : self._nq]):
            self._list_of_remap_objects[i](tracers[q], pe1, pe2, self._qs)

        if self._fill_negative_tracers is True:
            self._fillz(dp2, tracers)
