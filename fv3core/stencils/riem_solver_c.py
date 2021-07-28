import typing

from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval, log

import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.sim1_solver import Sim1Solver
from fv3core.utils.grid import GridIndexing
from fv3core.utils.typing import FloatField, FloatFieldIJ


@typing.no_type_check
def precompute(
    delpc: FloatField,
    cappa: FloatField,
    w3: FloatField,
    w: FloatField,
    gz: FloatField,
    dm: FloatField,
    q_con: FloatField,
    pem: FloatField,
    dz: FloatField,  # is actually delta of gz
    gm: FloatField,
    pm: FloatField,
    ptop: float,
):
    with computation(PARALLEL), interval(...):
        dm = delpc
        w = w3
    with computation(FORWARD):
        with interval(0, 1):
            pem = ptop
            peg = ptop
        with interval(1, None):
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
    with computation(PARALLEL), interval(0, -1):
        dz = gz[0, 0, 1] - gz
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cappa)
        dm /= constants.GRAV
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / log(peg[0, 0, 1] / peg)


def finalize(
    pe2: FloatField,
    pem: FloatField,
    hs: FloatFieldIJ,
    dz: FloatField,
    pef: FloatField,
    gz: FloatField,
    ptop: float,
):
    with computation(PARALLEL):
        with interval(0, 1):
            pef = ptop
        with interval(1, None):
            pef = pe2 + pem
    with computation(BACKWARD):
        with interval(-1, None):
            gz = hs
        with interval(0, -1):
            gz = gz[0, 0, 1] - dz * constants.GRAV


class RiemannSolverC:
    """
    Fortran subroutine Riem_Solver_C
    """

    def __init__(self, grid_indexing: GridIndexing, p_fac):
        origin = grid_indexing.origin_compute(add=(-1, -1, 0))
        domain = grid_indexing.domain_compute(add=(2, 2, 1))
        shape = grid_indexing.max_shape

        self._dm = utils.make_storage_from_shape(shape, origin)
        self._w = utils.make_storage_from_shape(shape, origin)
        self._pem = utils.make_storage_from_shape(shape, origin)
        self._pe = utils.make_storage_from_shape(shape, origin)
        self._gm = utils.make_storage_from_shape(shape, origin)
        self._dz = utils.make_storage_from_shape(shape, origin)
        self._pm = utils.make_storage_from_shape(shape, origin)

        self._precompute_stencil = FrozenStencil(
            precompute,
            origin=origin,
            domain=domain,
        )
        self._sim1_solve = Sim1Solver(
            p_fac,
            grid_indexing.isc - 1,
            grid_indexing.iec + 1,
            grid_indexing.jsc - 1,
            grid_indexing.jec + 1,
            grid_indexing.domain[2] + 1,
        )
        self._finalize_stencil = FrozenStencil(
            finalize,
            origin=origin,
            domain=domain,
        )

    def __call__(
        self,
        dt2: float,
        cappa: FloatField,
        ptop: float,
        hs: FloatFieldIJ,
        ws: FloatFieldIJ,
        ptc: FloatField,
        q_con: FloatField,
        delpc: FloatField,
        gz: FloatField,
        pef: FloatField,
        w3: FloatField,
    ):
        """
        Solves for the nonhydrostatic terms for vertical velocity (w)
        and non-hydrostatic pressure perturbation after C-grid winds advect
        and heights are updated.

        Args:
           dt2: acoustic timestep in seconds (in)
           cappa: ??? (in)
           ptop: pressure at top of atmosphere (in)
           hs: ??? (in)
           ws: vertical velocity of the lowest level (in)
           ptc: potential temperature (in)
           q_con: total condensate mixing ratio (in)
           delpc: vertical delta in pressure (in)
           gz: geopotential heigh (inout)
           pef: full hydrostatic pressure(inout)
           w3: vertical velocity (inout)
        """
        self._precompute_stencil(
            delpc,
            cappa,
            w3,
            self._w,
            gz,
            self._dm,
            q_con,
            self._pem,
            self._dz,
            self._gm,
            self._pm,
            ptop,
        )
        self._sim1_solve(
            dt2,
            self._gm,
            cappa,
            self._pe,
            self._dm,
            self._pm,
            self._pem,
            self._w,
            self._dz,
            ptc,
            ws,
        )
        self._finalize_stencil(self._pe, self._pem, hs, self._dz, pef, gz, ptop)
