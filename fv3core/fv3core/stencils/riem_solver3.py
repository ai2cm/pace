import math
import typing

from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    log,
)

import pace.dsl.gt4py_utils as utils
import pace.util.constants as constants
from fv3core._config import RiemannConfig
from fv3core.stencils.sim1_solver import Sim1Solver
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ


@typing.no_type_check
def precompute(
    delp: FloatField,
    cappa: FloatField,
    pe: FloatField,
    pe_init: FloatField,
    dm: FloatField,
    zh: FloatField,
    q_con: FloatField,
    pem: FloatField,
    peln: FloatField,
    pk3: FloatField,
    gm: FloatField,
    dz: FloatField,
    pm: FloatField,
    ptop: float,
    peln1: float,
    ptk: float,
):
    with computation(PARALLEL), interval(...):
        dm = delp
        pe_init = pe
    with computation(FORWARD):
        with interval(0, 1):
            pem = ptop
            peln = peln1
            pk3 = ptk
            peg = ptop
            pelng = peln1
        with interval(1, None):
            # TODO consolidate with riem_solver_c, same functions, math functions
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            peln = log(pem)
            # Excluding contribution from condensates
            # peln used during remap; pk3 used only for p_grad
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
            pelng = log(peg)
            # interface pk is using constant akap
            pk3 = exp(constants.KAPPA * peln)
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cappa)
        dm = dm * constants.RGRAV
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / (pelng[0, 0, 1] - pelng)
        dz = zh[0, 0, 1] - zh


def finalize(
    zs: FloatFieldIJ,
    dz: FloatField,
    zh: FloatField,
    peln_run: FloatField,
    peln: FloatField,
    pk3: FloatField,
    pk: FloatField,
    pem: FloatField,
    pe: FloatField,
    ppe: FloatField,
    pe_init: FloatField,
    last_call: bool,
):
    from __externals__ import beta, use_logp

    with computation(PARALLEL), interval(...):
        if __INLINED(use_logp):
            pk3 = peln_run
        if __INLINED(beta < -0.1):
            ppe = pe + pem
        else:
            ppe = pe
        if last_call:
            peln = peln_run
            pk = pk3
            pe = pem
        else:
            pe = pe_init
    with computation(BACKWARD):
        with interval(-1, None):
            zh = zs
        with interval(0, -1):
            zh = zh[0, 0, 1] - dz


class RiemannSolver3:
    """
    Fortran subroutine Riem_Solver3
    """

    def __init__(self, stencil_factory: StencilFactory, config: RiemannConfig):
        grid_indexing = stencil_factory.grid_indexing
        self._sim1_solve = Sim1Solver(
            stencil_factory,
            config.p_fac,
            grid_indexing.isc,
            grid_indexing.iec,
            grid_indexing.jsc,
            grid_indexing.jec,
            grid_indexing.domain[2] + 1,
        )
        if config.a_imp <= 0.999:
            raise NotImplementedError("a_imp <= 0.999 is not implemented")

        def make_storage():
            return utils.make_storage_from_shape(
                grid_indexing.max_shape,
                origin=grid_indexing.origin_compute(),
                backend=stencil_factory.backend,
            )

        self._tmp_dm = make_storage()
        self._tmp_pe_init = make_storage()
        self._tmp_pm = make_storage()
        self._tmp_pem = make_storage()
        self._tmp_peln_run = make_storage()
        self._tmp_gm = make_storage()

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
        wsd: FloatField,
        delz: FloatField,
        q_con: FloatField,
        delp: FloatField,
        pt: FloatField,
        zh: FloatField,
        pe: FloatField,
        ppe: FloatField,
        pk3: FloatField,
        pk: FloatField,
        peln: FloatField,
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
           last_call: boolean, is last acoustic timestep (in)
           dt: acoustic timestep in seconds (in)
           cappa: (in)
           ptop: pressure at top of atmosphere (in)
           zs: surface geopotential height(in)
           wsd: vertical velocity of the lowest level (in)
           delz: vertical delta of atmospheric layer in meters (in)
           q_con: total condensate mixing ratio (in)
           delp: vertical delta in pressure (in)
           pt: potential temperature (in)
           zh: geopotential height (inout)
           pe: full hydrostatic pressure(inout)
           ppe: non-hydrostatic pressure perturbation (inout)
           pk3: interface pressure raised to power of kappa using constant kappa (inout)
           pk: interface pressure raised to power of kappa, final acoustic value (inout)
           peln: logarithm of interface pressure(inout)
           w: vertical velocity (inout)
        """

        peln1 = math.log(ptop)
        ptk = math.exp(constants.KAPPA * peln1)
        self._precompute_stencil(
            delp,
            cappa,
            pe,
            self._tmp_pe_init,
            self._tmp_dm,
            zh,
            q_con,
            self._tmp_pem,
            self._tmp_peln_run,
            pk3,
            self._tmp_gm,
            delz,
            self._tmp_pm,
            ptop,
            peln1,
            ptk,
        )

        self._sim1_solve(
            dt,
            self._tmp_gm,
            cappa,
            pe,
            self._tmp_dm,
            self._tmp_pm,
            self._tmp_pem,
            w,
            delz,
            pt,
            wsd,
        )

        self._finalize_stencil(
            zs,
            delz,
            zh,
            self._tmp_peln_run,
            peln,
            pk3,
            pk,
            self._tmp_pem,
            pe,
            ppe,
            self._tmp_pe_init,
            last_call,
        )
