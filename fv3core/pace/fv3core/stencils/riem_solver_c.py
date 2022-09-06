import typing

from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval, log

import pace.dsl.gt4py_utils as utils
import pace.util.constants as constants
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core.stencils.sim1_solver import Sim1Solver


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
    """
    Args:
        delpc (in):
        cappa (in):
        w3 (in):
        w (out):
        gz (in):
        dm (out):
        q_con (in):
        pem (out):
        dz (out):
        gm (out):
        pm (out):
    """
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
    """
    Args:
        pe2 (in):
        pem (in):
        hs (in):
        dz (in):
        pef (out):
        gz (out):
    """
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

    def __init__(self, stencil_factory: StencilFactory, p_fac):
        grid_indexing = stencil_factory.grid_indexing
        origin = grid_indexing.origin_compute(add=(-1, -1, 0))
        domain = grid_indexing.domain_compute(add=(2, 2, 1))
        shape = grid_indexing.max_shape

        def make_storage():
            return utils.make_storage_from_shape(
                shape, origin, backend=stencil_factory.backend
            )

        self._dm = make_storage()
        self._w = make_storage()
        self._pem = make_storage()
        self._pe = make_storage()
        self._gm = make_storage()
        self._dz = make_storage()
        self._pm = make_storage()

        self._precompute_stencil = stencil_factory.from_origin_domain(
            precompute,
            origin=origin,
            domain=domain,
        )
        self._sim1_solve = Sim1Solver(
            stencil_factory,
            p_fac,
            grid_indexing.isc - 1,
            grid_indexing.iec + 1,
            grid_indexing.jsc - 1,
            grid_indexing.jec + 1,
            grid_indexing.domain[2] + 1,
        )
        self._finalize_stencil = stencil_factory.from_origin_domain(
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
           dt2 (in): acoustic timestep in seconds
           cappa (in): ???
           ptop (in): pressure at top of atmosphere
           hs (in): ???
           ws (in): vertical velocity of the lowest level
           ptc (in): potential temperature
           q_con (in): total condensate mixing ratio
           delpc (in): vertical delta in pressure
           gz (inout): geopotential height
           pef (out): full hydrostatic pressure
           w3 (in): vertical velocity
        """
        # TODO: this class is extremely similar in structure to RiemannSolver3,
        # can or should they be merged?
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
