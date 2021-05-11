from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, exp, interval, log

import fv3core._config as spec
import fv3core.utils.global_constants as constants
from fv3core.decorators import gtstencil
from fv3core.utils import Grid
from fv3core.utils.typing import FloatField


@gtscript.function
def set_cappa(qvapor, cvm, r_vir):
    cappa = constants.RDGAS / (constants.RDGAS + cvm / (1.0 + r_vir * qvapor))
    return cappa


@gtscript.function
def moist_cvm(qvapor, gz, ql, qs):
    cvm = (
        (1.0 - (qvapor + gz)) * constants.CV_AIR
        + qvapor * constants.CV_VAP
        + ql * constants.C_LIQ
        + qs * constants.C_ICE
    )
    return cvm


@gtscript.function
def moist_cv_nwat6_fn(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
):
    ql = qliquid + qrain
    qs = qice + qsnow + qgraupel
    gz = ql + qs
    cvm = moist_cvm(qvapor, gz, ql, qs)
    return cvm, gz


# TODO: Note untested
@gtscript.function
def moist_cv_nwat5_fn(qvapor, qliquid, qrain, qsnow, qice):
    ql = qliquid + qrain
    qs = qice + qsnow
    gz = ql + qs
    cvm = moist_cvm(qvapor, gz, ql, qs)
    return cvm, gz


# TODO: Note untested
@gtscript.function
def moist_cv_nwat4_fn(qvapor, qliquid, qrain):
    gz = qliquid + qrain
    cvm = (
        (1.0 - (qvapor + gz)) * constants.CV_AIR
        + qvapor * constants.CV_VAP
        + gz * constants.C_LIQ
    )
    return cvm, gz


# TODO: Note untested
@gtscript.function
def moist_cv_nwat3_fn(qvapor, qliquid, qice):
    gz = qliquid + qice
    cvm = moist_cvm(qvapor, gz, qliquid, qice)
    return cvm, gz


# TODO: Note untested
@gtscript.function
def moist_cv_nwat2_fn(qvapor, qliquid):
    qv = qvapor if qvapor > 0 else 0.0
    qs = qliquid if qliquid > 0 else 0.0
    gz = qs
    cvm = (1.0 - qv) * constants.CV_AIR + qv * constants.CV_VAP
    return cvm, gz


# TODO: Note untested
@gtscript.function
def moist_cv_nwat2_gfs_fn(qvapor, qliquid, t1):
    gz = qliquid if qliquid > 0 else 0.0
    qtmp = gz if t1 < constants.TICE - 15.0 else gz * (constants.TICE - t1) / 15.0
    qs = 0 if t1 > constants.TICE else qtmp
    ql = gz - qs
    qv = qvapor if qvapor > 0 else 0.0
    cvm = moist_cvm(qv, gz, ql, qs)
    return cvm, gz


# TODO: Note untested
@gtscript.function
def moist_cv_default_fn():
    gz = 0
    cvm = constants.CV_AIR
    return cvm, gz


@gtscript.function
def moist_pt_func(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: float,
):
    cvm, gz = moist_cv_nwat6_fn(
        qvapor, qliquid, qrain, qsnow, qice, qgraupel
    )  # if (nwat == 6) else moist_cv_default_fn(cv_air)
    q_con = gz
    cappa = set_cappa(qvapor, cvm, r_vir)
    pt = pt * exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
    return cvm, gz, q_con, cappa, pt


@gtstencil
def moist_pt(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: float,
):
    with computation(PARALLEL), interval(...):
        cvm, gz, q_con, cappa, pt = moist_pt_func(
            qvapor,
            qliquid,
            qrain,
            qsnow,
            qice,
            qgraupel,
            q_con,
            gz,
            cvm,
            pt,
            cappa,
            delp,
            delz,
            r_vir,
        )


@gtscript.function
def last_pt(
    pt: FloatField,
    dtmp: float,
    pkz: FloatField,
    gz: FloatField,
    qv: FloatField,
    zvir: float,
):
    return (pt + dtmp * pkz) / ((1.0 + zvir * qv) * (1.0 - gz))


def moist_pt_last_step(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    gz: FloatField,
    pt: FloatField,
    pkz: FloatField,
    dtmp: float,
    zvir: float,
):
    with computation(PARALLEL), interval(...):
        # if nwat == 2:
        #    gz = qliquid if qliquid > 0. else 0.
        #    qv = qvapor if qvapor > 0. else 0.
        #    pt = last_pt(pt, dtmp, pkz, gz, qv, zvir)
        # elif nwat == 6:
        gz = qliquid + qrain + qice + qsnow + qgraupel
        pt = last_pt(pt, dtmp, pkz, gz, qvapor, zvir)
        # else:
        #    cvm, gz = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel)
        #    pt = last_pt(pt, dtmp, pkz, gz, qvapor, zvir)


@gtscript.function
def compute_pkz_func(delp, delz, pt, cappa):
    # TODO use the exponential form for closer answer matching
    return exp(cappa * log(constants.RDG * delp / delz * pt))


@gtstencil
def moist_pkz(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    pkz: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: float,
):
    with computation(PARALLEL), interval(...):
        cvm, gz = moist_cv_nwat6_fn(
            qvapor, qliquid, qrain, qsnow, qice, qgraupel
        )  # if (nwat == 6) else moist_cv_default_fn(cv_air)
        q_con[0, 0, 0] = gz
        cappa = set_cappa(qvapor, cvm, r_vir)
        pkz = compute_pkz_func(delp, delz, pt, cappa)


def region_mode(j_2d: Optional[int], grid: Grid):
    if j_2d is None:
        origin = grid.compute_origin()
        domain = grid.domain_shape_compute()
        jslice = slice(grid.js, grid.je + 1)
    else:
        origin = (grid.is_, j_2d, 0)
        domain = (grid.nic, 1, grid.npz)
        jslice = slice(j_2d, j_2d + 1)
    return origin, domain, jslice


def compute_pt(
    qvapor_js: FloatField,
    qliquid_js: FloatField,
    qice_js: FloatField,
    qrain_js: FloatField,
    qsnow_js: FloatField,
    qgraupel_js: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: float,
    j_2d: int = None,
):
    origin, domain, _ = region_mode(j_2d, spec.grid)
    moist_pt(
        qvapor_js,
        qliquid_js,
        qrain_js,
        qsnow_js,
        qice_js,
        qgraupel_js,
        q_con,
        gz,
        cvm,
        pt,
        cappa,
        delp,
        delz,
        r_vir,
        origin=origin,
        domain=domain,
    )


def compute_pkz(
    qvapor_js: FloatField,
    qliquid_js: FloatField,
    qice_js: FloatField,
    qrain_js: FloatField,
    qsnow_js: FloatField,
    qgraupel_js: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    pkz: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: float,
    j_2d: int = None,
):
    grid = spec.grid
    origin, domain, _ = region_mode(j_2d, grid)

    moist_pkz(
        qvapor_js,
        qliquid_js,
        qrain_js,
        qsnow_js,
        qice_js,
        qgraupel_js,
        q_con,
        gz,
        cvm,
        pkz,
        pt,
        cappa,
        delp,
        delz,
        r_vir,
        origin=origin,
        domain=domain,
    )


def compute_last_step(
    pt: FloatField,
    pkz: FloatField,
    dtmp: FloatField,
    r_vir: float,
    qvapor: FloatField,
    qliquid: FloatField,
    qice: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    gz: FloatField,
):
    grid = spec.grid

    # Temporary Fix for calling moist_pt_last_step for verification tests
    last_step = gtstencil(moist_pt_last_step)

    last_step(
        qvapor,
        qliquid,
        qrain,
        qsnow,
        qice,
        qgraupel,
        gz,
        pt,
        pkz,
        dtmp,
        r_vir,
        origin=(grid.is_, grid.js, 0),
        domain=(grid.nic, grid.njc, grid.npz + 1),
    )


@gtstencil
def fvsetup_stencil(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    cvm: FloatField,
    pkz: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    dp1: FloatField,
    zvir: float,
    nwat: int,
    moist_phys: bool,
):
    with computation(PARALLEL), interval(...):
        # TODO: The conditional with gtscript function triggers and undefined
        # temporary variable, even though there are no new temporaries
        # if moist_phys:
        cvm, q_con = moist_cv_nwat6_fn(
            qvapor, qliquid, qrain, qsnow, qice, qgraupel
        )  # if (nwat == 6) else moist_cv_default_fn(cv_air)
        dp1 = zvir * qvapor
        cappa = constants.RDGAS / (constants.RDGAS + cvm / (1.0 + dp1))
        pkz = exp(
            cappa * log(constants.RDG * delp * pt * (1.0 + dp1) * (1.0 - q_con) / delz)
        )
        # else:
        #    dp1 = 0
        #    pkz = exp(constants.KAPPA * log(constants.RDG * delp * pt / delz)
        #


def fv_setup(
    pt: FloatField,
    pkz: FloatField,
    delz: FloatField,
    delp: FloatField,
    cappa: FloatField,
    q_con: FloatField,
    zvir: float,
    qvapor: FloatField,
    qliquid: FloatField,
    qice: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    cvm: FloatField,
    dp1: FloatField,
):
    if not spec.namelist.moist_phys:
        raise Exception("fvsetup is only implem ented for moist_phys=true")
    fvsetup_stencil(
        qvapor,
        qliquid,
        qrain,
        qsnow,
        qice,
        qgraupel,
        q_con,
        cvm,
        pkz,
        pt,
        cappa,
        delp,
        delz,
        dp1,
        zvir,
        spec.namelist.nwat,
        spec.namelist.moist_phys,
        origin=spec.grid.compute_origin(),
        domain=spec.grid.domain_shape_compute(),
    )
