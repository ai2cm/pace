from typing import Dict

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.fillz import FillNegativeTracerValues
from fv3core.stencils.map_single import MapSingle
from fv3core.utils.typing import FloatField


class MapNTracer:
    """
    Fortran code is mapn_tracer, test class is MapN_Tracer_2d
    """

    def __init__(self, kord: int, nq: int, i1: int, i2: int, j1: int, j2: int):
        grid = spec.grid
        namelist = spec.namelist
        self._origin = (i1, j1, 0)
        self._domain = ()
        self._nk = grid.npz
        self._nq = nq
        self._i1 = i1
        self._i2 = i2
        self._j1 = j1
        self._j2 = j2
        self._qs = utils.make_storage_from_shape(
            (grid.npx, grid.npy, self._nk + 1), origin=(0, 0, 0)
        )

        kord_tracer = [kord] * self._nq
        kord_tracer[5] = 9  # qcld

        self._list_of_remap_objects = [
            MapSingle(kord_tracer[i], 0, i1, i2, j1, j2)
            for i in range(len(kord_tracer))
        ]

        if namelist.fill:
            self._fill_negative_tracers = True
            self._fillz = FillNegativeTracerValues(
                self._list_of_remap_objects[0].i_extent,
                self._list_of_remap_objects[0].j_extent,
                self._nk,
                self._nq,
            )
        else:
            self._fill_negative_tracers = False

    def __call__(
        self,
        pe1: FloatField,
        pe2: FloatField,
        dp2: FloatField,
        tracers: Dict[str, "FloatField"],
        q_min: float,
    ):
        """
        Remaps the tracer species onto the Eulerian grid
        and optionally fills negative values in the tracer fields

        Args:
            pe1 (in): Lagrangian pressure levels
            pe2 (out): Eulerian pressure levels
            dp2 (in): Difference in pressure between Eulerian levels
            qs (out): Field to be remapped on deformed grid
            jfirst: Starting index of the J-dir compute domain
            jlast: Final index of the J-dir compute domain
        """
        for i, q in enumerate(utils.tracer_variables[0 : self._nq]):
            self._list_of_remap_objects[i](tracers[q], pe1, pe2, self._qs)

        if self._fill_negative_tracers is True:
            self._fillz(dp2, tracers)
