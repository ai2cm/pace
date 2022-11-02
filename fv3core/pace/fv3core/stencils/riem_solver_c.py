import typing

from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval, log

import pace.util
import pace.util.constants as constants
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core.stencils.sim1_solver import Sim1Solver
from pace.util import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM


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
        dm (out): delta mass, mass of gridcell per unit area
        q_con (in):
        pem (out): total hydrostatic pressure defined on interface
            (including condensate)
        dz (out):
        gm (out): gamma parameter, Cp/Cv, used to compute pressure gradient force
            using potential temperature and ideal gas law
        pm (out): hydrostatic cell mean pressure, derivation in documentation
            (Chapter 4? 7?)
            TODO: identify chapter reference, will be sent by Lucas
    """
    with computation(PARALLEL), interval(...):
        dm = delpc
        w = w3
    # peg is hydrostatic pressure defined on interface with condensate mass removed
    # pem is total hydrostatic pressure defined on interface (including condensate)
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
        # (1) From \partial p*/\partial z = -\rho g, we can separate and integrate
        # over a layer to get
        # \delta p* = -\bar\rho g \delta z = -\bar p* g\delta z / (R_d T_v)
        # (using the ideal gas law)
        # which we can solve for cell-mean pressure to get
        # \bar p* = - \delta p* \frac{R_d T_v}{g \delta z}

        # (2) From the hydrostatic balance, use the ideal gas law first to get
        # \partial p* / \partial z = - \frac{p* g}{R_d T_v}. Separating and integrating
        # yields
        # \delta log p* = - \frac{g\delta z}{R_d T_v}

        # The RHS of the final expression in (2) is the reciprocal of the fraction in
        # (1), giving us

        # \bar p* = \delta p* / \delta log p*
        # note log(b) - log(a) = log(b/a)
        pm = (peg[0, 0, 1] - peg) / (log(peg[0, 0, 1] / peg))


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
    Enforce vertical boundary conditions.

    The top of atmosphere pressure is constant.
    At bottom of domain, the height should be equal to hs, the surface elevation.

    Args:
        pe2 (in): nonhydrostatic perturbation pressure defined on interfaces
        pem (in): total hydrostatic pressure defined on interface (including condensate)
        hs (in): surface elevation
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

    Semi-implicit solver for pressure, vertical velocity, and dz (not a Riemann solver)

    accounts for:
    Vertically-propagating sound wave and straining terms
    vertical non-hydrostatic pressure gradient force
    change in layer interface heights due to straining/compression by sound waves
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        p_fac: float,
    ):
        grid_indexing = stencil_factory.grid_indexing
        origin = grid_indexing.origin_compute(add=(-1, -1, 0))
        domain = grid_indexing.domain_compute(add=(2, 2, 1))
        shape = grid_indexing.max_shape

        self._dm = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="kg")
        self._w = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="m/s")
        self._pem = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], units="Pa")
        self._pe = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], units="Pa")
        self._gm = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="")
        self._dz = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="m")
        self._pm = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="Pa")

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
           hs (in): surface height in m
           ws (in): vertical velocity of the lowest level
           ptc (in): potential temperature
           q_con (in): total condensate mixing ratio
           delpc (in): vertical delta in pressure
           gz (inout): geopotential height
           pef (out): full hydrostatic pressure
           w3 (in): vertical velocity
        """

        # TODO: integrate these notes into comments/code, double-check:
        """
        pe:interface full hydrostatic pressure, pe=pem at last_call
        pkc: (ppe, pe2 from SIM1_solver) interface non-hydrostatic
            pressure perturbation
        pem: interface full hydrostatic pressure, pem(i,k) = pem(i,k-1) + dm(i,k-1)
        peln2=log(pem)
        peg: as pem but without condensation,
            peg(i,k) = peg(i,k-1) + dm(i,k-1)*(1.-q_con(i,j,k-1))
        pm2: layer-mean full hydrostatic pressure without condensation
        pk3: interface pk with constant akap (p**k),
            pk3(i,j,k) = exp(akap*peln2(i,k))
        pe2: calculated from SIM1_sol
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
        # pe is nonhydrostatic perturbation pressure defined on interfaces
        self._finalize_stencil(self._pe, self._pem, hs, self._dz, pef, gz, ptop)
