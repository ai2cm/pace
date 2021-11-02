from fv3core.utils.typing import FloatField, Float
from gt4py.gtscript import (
    PARALLEL,
    FORWARD,
    BACKWARD,
    computation,
    horizontal,
    interval,
)

# TODO: we may not need to copy all or any of these
# [TODO] using a copy here because variables definition change inside physics
def copy_fields_in(     
    qvapor_in: FloatField,
    qliquid_in: FloatField,
    qrain_in: FloatField,
    qice_in: FloatField,
    qsnow_in: FloatField,
    qgraupel_in: FloatField,
    qo3mr_in: FloatField,
    qsgs_tke_in: FloatField,
    qcld_in: FloatField,
    pt_in: FloatField,
    delp_in: FloatField,
    delz_in: FloatField,
    ua_in: FloatField,
    va_in: FloatField,
    w_in: FloatField,
    omga_in: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qo3mr: FloatField,
    qsgs_tke: FloatField,
    qcld: FloatField,
    pt: FloatField,
    delp: FloatField,
    delz: FloatField,
    ua: FloatField,
    va: FloatField,
    w: FloatField,
    omga: FloatField
):
    with computation(PARALLEL), interval(...):
        qvapor=qvapor_in
        qliquid=qliquid_in
        qrain=qrain_in
        qsnow=qsnow_in
        qice=qice_in
        qgraupel=qgraupel_in
        qo3mr=qo3mr_in
        qsgs_tke=qsgs_tke_in
        qcld=qcld_in
        pt=pt_in
        delp=delp_in
        delz=delz_in
        ua=ua_in
        va=va_in
        w=w_in
        omga=omga_in
        
def fill_gfs(pe: FloatField, q: FloatField, q_min: Float):

    with computation(BACKWARD), interval(0, -3):
        if q[0, 0, 1] < q_min:
            q = q[0, 0, 0] + (q[0, 0, 1] - q_min) * (pe[0, 0, 2] - pe[0, 0, 1]) / (
                pe[0, 0, 1] - pe[0, 0, 0]
            )

    with computation(BACKWARD), interval(1, -3):
        if q[0, 0, 0] < q_min:
            q = q_min

    with computation(FORWARD), interval(1, -2):
        if q[0, 0, -1] < 0.0:
            q = q[0, 0, 0] + q[0, 0, -1] * (pe[0, 0, 0] - pe[0, 0, -1]) / (
                pe[0, 0, 1] - pe[0, 0, 0]
            )

    with computation(FORWARD), interval(0, -2):
        if q[0, 0, 0] < 0.0:
            q = 0.0


def prepare_tendencies_and_update_tracers(
    u_dt: FloatField,
    v_dt: FloatField,
    pt_dt: FloatField,
    u_t1: FloatField,
    v_t1: FloatField,
    pt_t1: FloatField,
    qvapor_t1: FloatField,
    qliquid_t1: FloatField,
    qrain_t1: FloatField,
    qsnow_t1: FloatField,
    qice_t1: FloatField,
    qgraupel_t1: FloatField,
    u_t0: FloatField,
    v_t0: FloatField,
    pt_t0: FloatField,
    qvapor_t0: FloatField,
    qliquid_t0: FloatField,
    qrain_t0: FloatField,
    qsnow_t0: FloatField,
    qice_t0: FloatField,
    qgraupel_t0: FloatField,
    prsi: FloatField,
    delp: FloatField,
    rdt: Float,
):
    """Gather tendencies and adjust dycore tracers values
    GFS total air mass = dry_mass + water_vapor (condensate excluded)
    GFS mixing ratios  = tracer_mass / (dry_mass + vapor_mass)
    FV3 total air mass = dry_mass + [water_vapor + condensate ]
    FV3 mixing ratios  = tracer_mass / (dry_mass+vapor_mass+cond_mass)
    """
    with computation(PARALLEL), interval(0, -1):
        u_dt += (u_t1 - u_t0) * rdt
        v_dt += (v_t1 - v_t0) * rdt
        pt_dt += (pt_t1 - pt_t0) * rdt
        dp = prsi[0, 0, 1] - prsi[0, 0, 0]
        qwat_qv = dp * qvapor_t1
        qwat_ql = dp * qliquid_t1
        qwat_qr = dp * qrain_t1
        qwat_qs = dp * qsnow_t1
        qwat_qi = dp * qice_t1
        qwat_qg = dp * qgraupel_t1
        qt = qwat_qv + qwat_ql + qwat_qr + qwat_qs + qwat_qi + qwat_qg
        q_sum = qvapor_t0 + qliquid_t0 + qrain_t0 + qsnow_t0 + qice_t0 + qgraupel_t0
        q0 = delp * (1.0 - q_sum) + qt
        delp = q0
        qvapor_t0 = qwat_qv / q0
        qliquid_t0 = qwat_ql / q0
        qrain_t0 = qwat_qr / q0
        qsnow_t0 = qwat_qs / q0
        qice_t0 = qwat_qi / q0
        qgraupel_t0 = qwat_qg / q0
