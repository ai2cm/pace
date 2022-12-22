import math
import typing

from gt4py.cartesian.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    log,
)

import pace.util
import pace.util.constants as constants
from pace.dsl.dace import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core._config import RiemannConfig
from pace.fv3core.stencils.sim1_solver import Sim1Solver
from pace.util import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM


@typing.no_type_check
def precompute(
    delp: FloatField,
    cappa: FloatField,
    pe: FloatField,
    pe_init: FloatField,
    delta_mass: FloatField,
    zh: FloatField,
    q_con: FloatField,
    p_interface: FloatField,
    log_p_interface: FloatField,
    pk3: FloatField,
    gamma: FloatField,
    dz: FloatField,
    p_gas: FloatField,
    ptop: float,
    peln1: float,
    ptk: float,
):
    """
    Args:
        delp (in): pressure thickness of atmospheric layer (Pa)
        cappa (in):
        pe (in):
        pe_init (out):
        delta_mass (out): mass thickness of atmospheric layer
        zh (in):
        q_con (in):
        p_interface (out): pressure defined on vertical interfaces (Pa)
        log_p_interface (out): log(pressure) defined on vertical interfaces
        pk3 (out):
        gamma (out):
        dz (out):
        p_gas (out): pressure defined at vertical mid levels due to gas-phase
            only, excluding condensates (Pa)
    """
    with computation(PARALLEL), interval(...):
        delta_mass = delp
        pe_init = pe
    with computation(FORWARD):
        with interval(0, 1):
            p_interface = ptop
            log_p_interface = peln1
            pk3 = ptk
            p_interface_gas = ptop
            log_p_interface_gas = peln1
        with interval(1, None):
            # TODO consolidate with riem_solver_c, same functions, math functions
            p_interface = p_interface[0, 0, -1] + delta_mass[0, 0, -1]
            log_p_interface = log(p_interface)
            # Excluding contribution from condensates
            # peln used during remap; pk3 used only for p_grad
            p_interface_gas = p_interface_gas[0, 0, -1] + delta_mass[0, 0, -1] * (
                1.0 - q_con[0, 0, -1]
            )
            log_p_interface_gas = log(p_interface_gas)
            # interface pk is using constant akap
            pk3 = exp(constants.KAPPA * log_p_interface)
    with computation(PARALLEL), interval(...):
        gamma = 1.0 / (1.0 - cappa)  # gamma, cp/cv
        delta_mass = delta_mass * constants.RGRAV
    with computation(PARALLEL), interval(0, -1):
        p_gas = (p_interface_gas[0, 0, 1] - p_interface_gas) / (
            log_p_interface_gas[0, 0, 1] - log_p_interface_gas
        )
        dz = zh[0, 0, 1] - zh


def finalize(
    zs: FloatFieldIJ,
    dz: FloatField,
    zh: FloatField,
    log_p_interface_internal: FloatField,
    log_p_interface_out: FloatField,
    pk3: FloatField,
    pk: FloatField,
    p_interface: FloatField,
    pe: FloatField,
    ppe: FloatField,
    pe_init: FloatField,
    last_call: bool,
):
    """
    Updates auxilary pressure values

    Args:
        zs (in):
        dz (in):
        zh (out):
        peln_run (in): log(pressure) defined on vertical interfaces, as used
            for computation in this module
        peln (out): log(pressure) defined on vertical interfaces, memory
            to be returned to calling module
        pk3 (in):
        pk (out):
        p_interface (in):
        pe (inout):
        ppe (out):
        pe_init (in):
        last_call (in):
    """
    from __externals__ import beta, use_logp

    with computation(PARALLEL), interval(...):
        if __INLINED(use_logp):
            pk3 = log_p_interface_internal
        if __INLINED(beta < -0.1):
            ppe = pe + p_interface
        else:
            ppe = pe
        if last_call:
            log_p_interface_out = log_p_interface_internal
            pk = pk3
            pe = p_interface
        else:
            pe = pe_init
    with computation(BACKWARD):
        with interval(-1, None):
            zh = zs
        with interval(0, -1):
            zh = zh[0, 0, 1] - dz


class NonhydrostaticVerticalSolver:
    """
    Fortran subroutine Riem_Solver3

    Like RiemannSolverC, but for the d-grid.

    Difference is that this uses the advanced values for the d-grid full timestep,
    while RiemannSolverC uses the half time stepped c-grid w, delp, and gz.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        config: RiemannConfig,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._sim1_solve = Sim1Solver(
            stencil_factory,
            config.p_fac,
            n_halo=0,
        )
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )

        if config.a_imp <= 0.999:
            raise NotImplementedError("a_imp <= 0.999 is not implemented")

        self._delta_mass = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="kg")
        self._tmp_pe_init = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM], units="Pa"
        )
        self._p_gas = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="Pa")
        self._p_interface = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM], units="Pa"
        )
        self._log_p_interface = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM], units="log(Pa)"
        )

        # gamma parameter is (cp/cv)
        self._gamma = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="")

        riemorigin = grid_indexing.origin_compute()
        domain = grid_indexing.domain_compute(add=(0, 0, 1))
        self._precompute_stencil = stencil_factory.from_origin_domain(
            precompute,
            origin=riemorigin,
            domain=domain,
        )
        self._finalize_stencil = stencil_factory.from_origin_domain(
            finalize,
            externals={"use_logp": config.use_logp, "beta": config.beta},
            origin=riemorigin,
            domain=domain,
        )

    def __call__(
        self,
        last_call: bool,
        dt: float,
        cappa: FloatField,
        ptop: float,
        zs: FloatFieldIJ,
        ws: FloatFieldIJ,
        delz: FloatField,
        q_con: FloatField,
        delp: FloatField,
        pt: FloatField,
        zh: FloatField,
        p: FloatField,
        ppe: FloatField,
        pk3: FloatField,
        pk: FloatField,
        log_p_interface: FloatField,
        w: FloatFieldIJ,
    ):
        """
        Solves for the nonhydrostatic terms for vertical velocity (w)
        and non-hydrostatic pressure perturbation after D-grid winds advect
        and heights are updated.
        This accounts for vertically propagating sound waves. Currently
        the only implemented option of a_imp > 0.999 calls a semi-implicit
        method solver, and the exact Riemann solver best used for > 1km resolution
        simulations is not yet implemented.

        Args:
            last_call (in): boolean, is last acoustic timestep
            dt (in): acoustic timestep in seconds
            cappa (in):
            ptop (in): pressure at top of atmosphere
            zs (in): surface geopotential height
            ws (in): surface vertical wind (e.g. due to topography)
            delz (inout): vertical delta of atmospheric layer in meters
            q_con (in): total condensate mixing ratio
            delp (in): vertical delta in pressure
            pt (in): potential temperature
            zh (inout): geopotential height
            p (inout): full hydrostatic pressure
            ppe (out): non-hydrostatic pressure perturbation
            pk3 (inout): interface pressure raised to power of kappa
                using constant kappa
            pk (out): interface pressure raised to power of kappa, final acoustic value
            log_p_interface (out): logarithm of interface pressure,
                only written if last_call=True
            w (inout): vertical velocity
        """

        # TODO: propagate variable renaming for these into stencils here and
        # in Sim1Solver
        # temporaries:
        # cp2 is cappa copied into a 2d variable, copied for vectorization reasons
        #     we should be able to remove this as gt4py optimizes automatically
        # peln2 is peln copied into a 2d variable, copied for vectorization reasons
        #     we should be able to remove this as gt4py optimizes automatically
        # pk3 is p**kappa
        # pm is layer-mean hydrostatic pressure due to gas phase
        #       (with condensates removed)
        # gm2 is gamma (cp/cv)
        # dz2 is delz

        peln1 = math.log(ptop)
        # ptk = ptop ** kappa
        ptk = math.exp(constants.KAPPA * peln1)

        self._precompute_stencil(
            delp,
            cappa,
            p,
            self._tmp_pe_init,
            self._delta_mass,
            zh,
            q_con,
            self._p_interface,
            self._log_p_interface,
            pk3,
            self._gamma,
            delz,
            self._p_gas,
            ptop,
            peln1,
            ptk,
        )

        self._sim1_solve(
            dt,
            self._gamma,
            cappa,
            p,
            self._delta_mass,
            self._p_gas,
            self._p_interface,
            w,
            delz,
            pt,
            ws,
        )

        self._finalize_stencil(
            zs,
            delz,
            zh,
            self._log_p_interface,
            log_p_interface,
            pk3,
            pk,
            self._p_interface,
            p,
            ppe,
            self._tmp_pe_init,
            last_call,
        )
