import math

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

import fv3core._config as spec
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.sim1_solver import Sim1Solver
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtstencil()
def precompute(
    delp: FloatField,
    cappa: FloatField,
    pe: FloatField,
    pe_init: FloatField,
    cp3: FloatField,
    dm: FloatField,
    zh: FloatField,
    q_con: FloatField,
    pem: FloatField,
    peln: FloatField,
    pk3: FloatField,
    peg: FloatField,
    pelng: FloatField,
    gm: FloatField,
    dz: FloatField,
    pm: FloatField,
    ptop: float,
    peln1: float,
    ptk: float,
    rgrav: float,
    akap: float,
):
    with computation(PARALLEL), interval(...):
        dm = delp
        cp3 = cappa
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
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
            pelng = log(peg)
            pk3 = exp(akap * peln)
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cp3)
        dm = dm * rgrav
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / (pelng[0, 0, 1] - pelng)
        dz = zh[0, 0, 1] - zh


@gtstencil()
def last_call_copy(
    peln_run: FloatField,
    peln: FloatField,
    pk3: FloatField,
    pk: FloatField,
    pem: FloatField,
    pe: FloatField,
):
    with computation(PARALLEL), interval(...):
        peln = peln_run
        pk = pk3
        pe = pem


@gtstencil()
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
    with computation(PARALLEL), interval(...):
        if __INLINED(spec.namelist.use_logp):
            pk3 = peln_run
        if __INLINED(spec.namelist.beta < -0.1):
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


def compute(
    last_call: bool,
    dt: float,
    akap: float,
    cappa: FloatField,
    ptop: float,
    zs: FloatFieldIJ,
    w: FloatField,
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
    wsd: FloatFieldIJ,
):
    grid = spec.grid
    rgrav = 1.0 / constants.GRAV
    km = grid.npz - 1
    peln1 = math.log(ptop)
    ptk = math.exp(akap * peln1)
    shape = w.shape
    domain = (grid.nic, grid.njc, km + 2)
    riemorigin = (grid.is_, grid.js, 0)

    dm = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem3_dm")
    cp3 = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem3_cp3")
    pe_init = utils.make_storage_from_shape(
        shape, riemorigin, cache_key="riem3_pe_init"
    )
    pm = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem_solver3_pm")
    pem = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem_solver3_pem")
    peln_run = utils.make_storage_from_shape(
        shape, riemorigin, cache_key="riem_solver3_peln_run"
    )
    peg = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem_solver3_peg")
    pelng = utils.make_storage_from_shape(
        shape, riemorigin, cache_key="riem_solver3_pelng"
    )
    gm = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem_solver3_gm")

    precompute(
        delp,
        cappa,
        pe,
        pe_init,
        cp3,
        dm,
        zh,
        q_con,
        pem,
        peln_run,
        pk3,
        peg,
        pelng,
        gm,
        delz,
        pm,
        ptop,
        peln1,
        ptk,
        rgrav,
        akap,
        origin=riemorigin,
        domain=domain,
    )

    sim1_solve = utils.cached_stencil_class(Sim1Solver)(
        spec.namelist,
        spec.grid,
        grid.is_,
        grid.ie,
        grid.js,
        grid.je,
        cache_key="riem_solver3_sim1solver",
    )
    sim1_solve(dt, gm, cp3, pe, dm, pm, pem, w, delz, pt, wsd)

    finalize(
        zs,
        delz,
        zh,
        peln_run,
        peln,
        pk3,
        pk,
        pem,
        pe,
        ppe,
        pe_init,
        last_call,
        origin=riemorigin,
        domain=domain,
    )
