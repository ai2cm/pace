import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, exp, interval, log

import fv3core._config as spec
import fv3core.utils.global_constants as constants
from fv3core.decorators import gtstencil
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


@gtstencil()
def moist_cv_nwat6(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    cvm: FloatField,
):
    with computation(PARALLEL), interval(...):
        cvm = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel)


@gtscript.function
def te_always_part(u, v, w, phis, rsin2, cosa_s):
    return 0.5 * (
        phis
        + phis[0, 0, 1]
        + w ** 2
        + 0.5
        * rsin2
        * (
            u ** 2
            + u[0, 1, 0] ** 2
            + v ** 2
            + v[1, 0, 0] ** 2
            - (u + u[0, 1, 0]) * (v + v[1, 0, 0]) * cosa_s
        )
    )


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


@gtstencil()
def moist_te_2d(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    te_2d: FloatField,
    delp: FloatField,
    pt: FloatField,
    phis: FloatField,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    rsin2: FloatField,
    cosa_s: FloatField,
    r_vir: float,
):
    with computation(FORWARD):
        with interval(0, 1):
            cvm, gz = moist_cv_nwat6_fn(
                qvapor, qliquid, qrain, qsnow, qice, qgraupel
            )  # if (nwat == 6) else moist_cv_default_fn()
            q_con = gz
            te_2d = te_2d + delp * (
                cvm * pt / ((1.0 + r_vir * qvapor) * (1.0 - gz))
                + te_always_part(u, v, w, phis, rsin2, cosa_s)
            )
        with interval(1, None):
            cvm, gz = moist_cv_nwat6_fn(
                qvapor, qliquid, qrain, qsnow, qice, qgraupel
            )  # if (nwat == 6) else moist_cv_default_fn()
            q_con = gz
            te_2d = te_2d[0, 0, -1] + delp * (
                cvm * pt / ((1.0 + r_vir * qvapor) * (1.0 - gz))
                + te_always_part(u, v, w, phis, rsin2, cosa_s)
            )


# TODO: Calling gtscript functions from inside the if statements is causing
# problems, if we want 'moist_phys' to be changeable, we either need to
# duplicate the stencil code or fix the gt4py bug.
@gtstencil()
def moist_te_total_energy(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    te_2d: FloatField,
    delp: FloatField,
    pt: FloatField,
    phis: FloatField,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    rsin2: FloatField,
    cosa_s: FloatField,
    delz: FloatField,
    r_vir: float,
    moist_phys: bool,
):
    with computation(BACKWARD):
        with interval(-1, None):
            phiz = phis
        with interval(0, -1):
            phiz = phiz[0, 0, 1] - constants.GRAV * delz
            te_2d = 0.0
    with computation(FORWARD), interval(0, -1):
        qd = 0.0
        cvm = 0.0
        # if moist_phys:
        cvm, qd = moist_cv_nwat6_fn(
            qvapor, qliquid, qrain, qsnow, qice, qgraupel
        )  # if (nwat == 6) else moist_cv_default_fn()
        te_2d = te_2d[0, 0, -1] + delp * (
            cvm * pt + te_always_part(u, v, w, phiz, rsin2, cosa_s)
        )
        # else:
        #    te_2d = te_2d[0, 0, -1] + delp * (
        #        constants.CV_AIR * pt + te_always_part(u, v, w, phiz, rsin2, cosa_s)
        #    )


@gtstencil()
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


@gtstencil()
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


@gtstencil()
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


def region_mode(j_2d, grid):
    if j_2d is None:
        origin = grid.compute_origin()
        domain = grid.domain_shape_compute()
        jslice = slice(grid.js, grid.je + 1)
    else:
        origin = (grid.is_, j_2d, 0)
        domain = (grid.nic, 1, grid.npz)
        jslice = slice(j_2d, j_2d + 1)
    return origin, domain, jslice


# Computes the FV3-consistent moist heat capacity under constant volume,
# including the heating capacity of water vapor and condensates.
# See emanuel1994atmospheric for information on variable heat capacities.

# assumes 3d variables are indexed to j
def compute_te(
    qvapor_js: FloatField,
    qliquid_js: FloatField,
    qice_js: FloatField,
    qrain_js: FloatField,
    qsnow_js: FloatField,
    qgraupel_js: FloatField,
    te_2d: FloatField,
    gz: FloatField,
    cvm: FloatField,
    delp: FloatField,
    q_con: FloatField,
    pt: FloatField,
    phis: FloatField,
    w: FloatField,
    u: FloatField,
    v: FloatField,
    r_vir: float,
    j_2d: int = None,
):
    # TODO -- to do this cleanly, we probably need if blocks working inside stencils
    if spec.namelist.nwat != 6:
        raise Exception("We still need to implement other nwats for moist_cv")
    grid = spec.grid
    origin, domain, jslice = region_mode(j_2d, grid)
    moist_te_2d(
        qvapor_js,
        qliquid_js,
        qrain_js,
        qsnow_js,
        qice_js,
        qgraupel_js,
        q_con,
        gz,
        cvm,
        te_2d,
        delp,
        pt,
        phis,
        u,
        v,
        w,
        grid.rsin2,
        grid.cosa_s,
        r_vir,
        origin=origin,
        domain=domain,
    )


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
    grid = spec.grid
    origin, domain, jslice = region_mode(j_2d, grid)
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
    origin, domain, jslice = region_mode(j_2d, grid)

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


@gtstencil()
def compute_pkz_stencil_func(
    pkz: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    pt: FloatField,
):
    with computation(PARALLEL), interval(...):
        pkz = compute_pkz_func(delp, delz, pt, cappa)


def compute_total_energy(
    u: FloatField,
    v: FloatField,
    w: FloatField,
    delz: FloatField,
    pt: FloatField,
    delp: FloatField,
    qc: FloatField,
    pe: FloatField,
    peln: FloatField,
    hs: FloatField,
    zvir: float,
    te_2d: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qice: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
):
    grid = spec.grid
    if spec.namelist.hydrostatic:
        raise Exception("Porting compute_total_energy incomplete for hydrostatic=True")
    if not spec.namelist.moist_phys:
        raise Exception(
            "To run without moist_phys, the if conditional bug needs to be fixed, "
            "or code needs to be duplicated"
        )
    moist_te_total_energy(
        qvapor,
        qliquid,
        qrain,
        qsnow,
        qice,
        qgraupel,
        te_2d,
        delp,
        pt,
        hs,
        u,
        v,
        w,
        grid.rsin2,
        grid.cosa_s,
        delz,
        zvir,
        spec.namelist.nwat,
        spec.namelist.moist_phys,
        origin=(grid.is_, grid.js, 0),
        domain=(grid.nic, grid.njc, grid.npz + 1),
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
    moist_pt_last_step(
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


@gtstencil()
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
