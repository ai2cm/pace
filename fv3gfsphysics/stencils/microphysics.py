from gt4py.gtscript import (
    PARALLEL,
    FORWARD,
    BACKWARD,
    computation,
    horizontal,
    interval,
)
from fv3gfsphysics.utils.global_config import *
from fv3gfsphysics.utils.serialization import *
import gt4py.gtscript as gtscript
import math as mt
from fv3gfsphysics.utils.global_constants import *


@gtscript.stencil(backend=BACKEND)
def fields_init(
    land: FIELD_FLT,
    area: FIELD_FLT,
    h_var: FIELD_FLT,
    rh_adj: FIELD_FLT,
    rh_rain: FIELD_FLT,
    graupel: FIELD_FLT,
    ice: FIELD_FLT,
    rain: FIELD_FLT,
    snow: FIELD_FLT,
    qa: FIELD_FLT,
    qg: FIELD_FLT,
    qi: FIELD_FLT,
    ql: FIELD_FLT,
    qn: FIELD_FLT,
    qr: FIELD_FLT,
    qs: FIELD_FLT,
    qv: FIELD_FLT,
    pt: FIELD_FLT,
    delp: FIELD_FLT,
    dz: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    qi_dt: FIELD_FLT,
    qs_dt: FIELD_FLT,
    uin: FIELD_FLT,
    vin: FIELD_FLT,
    qa0: FIELD_FLT,
    qg0: FIELD_FLT,
    qi0: FIELD_FLT,
    ql0: FIELD_FLT,
    qr0: FIELD_FLT,
    qs0: FIELD_FLT,
    qv0: FIELD_FLT,
    t0: FIELD_FLT,
    dp0: FIELD_FLT,
    den0: FIELD_FLT,
    dz0: FIELD_FLT,
    u0: FIELD_FLT,
    v0: FIELD_FLT,
    dp1: FIELD_FLT,
    p1: FIELD_FLT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    ccn: FIELD_FLT,
    c_praut: FIELD_FLT,
    use_ccn: DTYPE_INT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    d0_vap: DTYPE_FLT,
    lv00: DTYPE_FLT,
    dt_in: DTYPE_FLT,
    rdt: DTYPE_FLT,
    cpaut: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        # Initialize precipitation
        graupel = 0.0
        rain = 0.0
        snow = 0.0
        ice = 0.0

        # This is to prevent excessive build-up of cloud ice from
        # external sources
        if de_ice == 1:

            qio = qi - dt_in * qi_dt  # Orginal qi before phys
            qin = max(qio, qi0_max)  # Adjusted value

            if qi > qin:

                qs = qs + qi - qin
                qi = qin

                dqi = (qin - qio) * rdt  # Modified qi tendency
                qs_dt = qs_dt + qi_dt - dqi
                qi_dt = dqi

        qiz = qi
        qsz = qs

        t0 = pt
        tz = t0
        dp1 = delp
        dp0 = dp1  # Moist air mass * grav

        # Convert moist mixing ratios to dry mixing ratios
        qvz = qv
        qlz = ql
        qrz = qr
        qgz = qg

        dp1 = dp1 * (1.0 - qvz)
        omq = dp0 / dp1

        qvz = qvz * omq
        qlz = qlz * omq
        qrz = qrz * omq
        qiz = qiz * omq
        qsz = qsz * omq
        qgz = qgz * omq

        qa0 = qa
        dz0 = dz

        den0 = -dp1 / (grav * dz0)  # Density of dry air
        p1 = den0 * rdgas * t0  # Dry air pressure

        # Save a copy of old values for computing tendencies
        qv0 = qvz
        ql0 = qlz
        qr0 = qrz
        qi0 = qiz
        qs0 = qsz
        qg0 = qgz

        # For sedi_momentum
        u0 = uin
        v0 = vin
        u1 = u0
        v1 = v0

        if prog_ccn == 1:

            # Convert #/cc to #/m^3
            ccn = qn * 1.0e6
            c_praut = cpaut * (ccn * rhor) ** (-1.0 / 3.0)

        else:

            ccn = (ccn_l * land + ccn_o * (1.0 - land)) * 1.0e6

    with computation(FORWARD):

        with interval(0, 1):

            if (prog_ccn == 0) and (use_ccn == 1):

                # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                ccn = ccn * rdgas * tz / p1
        with interval(1, None):

            if (prog_ccn == 0) and (use_ccn == 1):

                # Propagate downwards in the atmosphere previously computed values of ccn
                ccn = ccn[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if prog_ccn == 0:
            c_praut = cpaut * (ccn * rhor) ** (-1.0 / 3.0)

        # Calculate horizontal subgrid variability
        # Total water subgrid deviation in horizontal direction
        # Default area dependent form: use dx ~ 100 km as the base
        s_leng = sqrt(sqrt(area * 1.0e-10))
        t_land = dw_land * s_leng
        t_ocean = dw_ocean * s_leng
        h_var = t_land * land + t_ocean * (1.0 - land)
        h_var = min(0.2, max(0.01, h_var))

        # Relative humidity increment
        rh_adj = 1.0 - h_var - rh_inc
        rh_rain = max(0.35, rh_adj - rh_inr)

        # Fix all negative water species
        if fix_negative == 1:

            # Define heat capacity and latent heat coefficient
            cvm = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
            lcpk = (lv00 + d0_vap * tz) / cvm
            icpk = (li00 + dc_ice * tz) / cvm

            # Ice phase

            # If cloud ice < 0, borrow from snow
            if qiz < 0.0:

                qsz = qsz + qiz
                qiz = 0.0

            # If snow < 0, borrow from graupel
            if qsz < 0.0:

                qgz = qgz + qsz
                qsz = 0.0

            # If graupel < 0, borrow from rain
            if qgz < 0.0:

                qrz = qrz + qgz
                tz = tz - qgz * icpk  # Heating
                qgz = 0.0

            # Liquid phase

            # If rain < 0, borrow from cloud water
            if qrz < 0.0:

                qlz = qlz + qrz
                qrz = 0.0

            # If cloud water < 0, borrow from water vapor
            if qlz < 0.0:

                qvz = qvz + qlz
                tz = tz - qlz * lcpk  # Heating
                qlz = 0.0

    with computation(BACKWARD), interval(0, -1):

        # Fix water vapor; borrow from below
        if (fix_negative == 1) and (qvz[0, 0, 1] < 0.0):
            qvz[0, 0, 0] = qvz[0, 0, 0] + qvz[0, 0, 1] * dp1[0, 0, 1] / dp1[0, 0, 0]

    with computation(PARALLEL), interval(1, None):

        if (fix_negative == 1) and (qvz < 0.0):
            qvz = 0.0

    # Bottom layer; borrow from above
    with computation(PARALLEL):
        with interval(0, 1):

            flag = 0

            if (fix_negative == 1) and (qvz < 0.0) and (qvz[0, 0, 1] > 0.0):

                dq = min(-qvz[0, 0, 0] * dp1[0, 0, 0], qvz[0, 0, 1] * dp1[0, 0, 1])
                flag = 1
        with interval(1, 2):

            flag = 0

            if (fix_negative == 1) and (qvz[0, 0, -1] < 0.0) and (qvz > 0.0):

                dq = min(-qvz[0, 0, -1] * dp1[0, 0, -1], qvz[0, 0, 0] * dp1[0, 0, 0])
                flag = 1

    with computation(PARALLEL):
        with interval(0, 1):

            if flag == 1:

                qvz = qvz + dq / dp1
        with interval(1, 2):

            if flag == 1:

                qvz = qvz - dq / dp1


@gtscript.stencil(backend=BACKEND)
def warm_rain(
    h_var: FIELD_FLT,
    rain: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    den: FIELD_FLT,
    denfac: FIELD_FLT,
    w: FIELD_FLT,
    t0: FIELD_FLT,
    den0: FIELD_FLT,
    dz0: FIELD_FLT,
    dz1: FIELD_FLT,
    dp1: FIELD_FLT,
    m1: FIELD_FLT,
    vtrz: FIELD_FLT,
    ccn: FIELD_FLT,
    c_praut: FIELD_FLT,
    m1_sol: FIELD_FLT,
    m2_rain: FIELD_FLT,
    m2_sol: FIELD_FLT,
    is_first: DTYPE_INT,
    do_sedi_w: DTYPE_INT,
    p_nonhydro: DTYPE_INT,
    use_ccn: DTYPE_INT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    d0_vap: DTYPE_FLT,
    lv00: DTYPE_FLT,
    fac_rc: DTYPE_FLT,
    cracw: DTYPE_FLT,
    crevp_0: DTYPE_FLT,
    crevp_1: DTYPE_FLT,
    crevp_2: DTYPE_FLT,
    crevp_3: DTYPE_FLT,
    crevp_4: DTYPE_FLT,
    t_wfr: DTYPE_FLT,
    so3: DTYPE_FLT,
    dt_rain: DTYPE_FLT,
    zs: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        if is_first == 1:

            # Define air density based on hydrostatical property
            if p_nonhydro == 1:

                dz1 = dz0
                den = den0  # Dry air density remains the same
                denfac = sqrt(sfcrho / den)

            else:

                dz1 = dz0 * tz / t0  # Hydrostatic balance
                den = den0 * dz0 / dz1
                denfac = sqrt(sfcrho / den)

        # Time-split warm rain processes: 1st pass
        dt5 = 0.5 * dt_rain

        # Terminal speed of rain
        m1_rain = 0.0

    with computation(BACKWARD):
        with interval(-1, None):

            if qrz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        with interval(0, -1):

            if no_fall[0, 0, 1] == 1:

                if qrz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    with computation(FORWARD), interval(1, None):

        if no_fall[0, 0, -1] == 0:
            no_fall = no_fall[0, 0, -1]

    with computation(PARALLEL), interval(...):

        vtrz, r1 = compute_rain_fspeed(no_fall, qrz, den)

    with computation(FORWARD):

        with interval(0, 1):

            if no_fall == 0:
                ze = zs - dz1

        with interval(1, None):

            if no_fall == 0:
                ze = ze[0, 0, -1] - dz1  # dz < 0

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            # Evaporation and accretion of rain for the first 1/2 time step
            qgz, qiz, qlz, qrz, qsz, qvz, tz = revap_racc(
                dt5,
                c_air,
                c_vap,
                d0_vap,
                lv00,
                t_wfr,
                cracw,
                crevp_0,
                crevp_1,
                crevp_2,
                crevp_3,
                crevp_4,
                h_var,
                qgz,
                qiz,
                qlz,
                qrz,
                qsz,
                qvz,
                tz,
                den,
                denfac,
            )

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    # Mass flux induced by falling rain
    with computation(PARALLEL):
        with interval(0, 1):

            if (use_ppm == 1) and (no_fall == 0):

                zt = ze - dt5 * (vtrz[0, 0, 1] + vtrz)

                zt_kbot1 = zs - dt_rain * vtrz
        with interval(1, -1):

            if (use_ppm == 1) and (no_fall == 0):
                zt = ze - dt5 * (vtrz[0, 0, 1] + vtrz)
        with interval(-1, None):

            if (use_ppm == 1) and (no_fall == 0):
                zt = ze

    with computation(BACKWARD):

        with interval(1, -1):

            if (use_ppm == 1) and (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

        with interval(0, 1):

            if use_ppm == 1:

                if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                    zt = zt[0, 0, 1] - dz_min

                if (no_fall == 0) and (zt_kbot1 >= zt):
                    zt_kbot1 = zt - dz_min

    with computation(FORWARD), interval(1, None):

        if (use_ppm == 1) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, -1]

    with computation(PARALLEL):
        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dt_rain * vtrz
            qrz = qrz * dp1

    # Sedimentation
    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qrz / (dz + dd)

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qrz[0, 0, 0] + dd[0, 0, 1] * qm[0, 0, 1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_rain = qrz - qm

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_rain = m1_rain[0, 0, 1] + qrz[0, 0, 0] - qm

    with computation(FORWARD):

        with interval(0, 1):
            if (use_ppm == 0) and (no_fall == 0):
                r1 = m1_rain

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                r1 = r1[0, 0, -1]

    with computation(PARALLEL):
        with interval(0, -1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qrz = qm / dp1

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0]
                        - m1_rain[0, 0, 1] * vtrz[0, 0, 1]
                        + m1_rain * vtrz
                    ) / (dm + m1_rain[0, 0, 1] - m1_rain)
        with interval(-1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qrz = qm / dp1

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_rain * vtrz) / (dm - m1_rain)

    # Heat transportation during sedimentation
    with computation(PARALLEL):
        with interval(0, -1):

            if (do_sedi_heat == 1) and (no_fall == 0):

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )
        with interval(-1, None):

            if (do_sedi_heat == 1) and (no_fall == 0):

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )

                # - Assumption: The ke in the falling condensates is negligible
                #               compared to the potential energy that was
                #               unaccounted for. Local thermal equilibrium is
                #               assumed, and the loss in pe is transformed into
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tmp = cvn + m1_rain * c_liq
                tz = tz + m1_rain * dgz / tmp

    # Implicit algorithm
    with computation(BACKWARD), interval(0, -1):

        if (do_sedi_heat == 1) and (no_fall == 0):

            tz[0, 0, 0] = (
                (cvn + c_liq * (m1_rain - m1_rain[0, 0, 1])) * tz[0, 0, 0]
                + m1_rain[0, 0, 1] * c_liq * tz[0, 0, 1]
                + dgz * (m1_rain[0, 0, 1] + m1_rain)
            ) / (cvn + c_liq * m1_rain)

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            # Evaporation and accretion of rain for the remaining 1/2 time step
            qgz, qiz, qlz, qrz, qsz, qvz, tz = revap_racc(
                dt5,
                c_air,
                c_vap,
                d0_vap,
                lv00,
                t_wfr,
                cracw,
                crevp_0,
                crevp_1,
                crevp_2,
                crevp_3,
                crevp_4,
                h_var,
                qgz,
                qiz,
                qlz,
                qrz,
                qsz,
                qvz,
                tz,
                den,
                denfac,
            )

        # Auto-conversion assuming linear subgrid vertical distribution of
        # cloud water following lin et al. 1994, mwr
        if irain_f != 0:

            qlz, qrz = autoconv_no_subgrid_var(
                use_ccn, fac_rc, t_wfr, so3, dt_rain, qlz, qrz, tz, den, ccn, c_praut
            )

    # With subgrid variability
    with computation(BACKWARD):

        with interval(-1, None):

            if (irain_f == 0) and (z_slope_liq == 1):
                dl = 0.0

        with interval(0, -1):

            if (irain_f == 0) and (z_slope_liq == 1):
                dq = 0.5 * (qlz[0, 0, 0] - qlz[0, 0, 1])

    with computation(PARALLEL):
        with interval(0, 1):

            if (irain_f == 0) and (z_slope_liq == 1):
                dl = 0.0

        with interval(1, -1):

            if (irain_f == 0) and (z_slope_liq == 1):

                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                dl = 0.5 * min(abs(dq + dq[0, 0, -1]), 0.5 * qlz[0, 0, 0])

                if dq * dq[0, 0, -1] <= 0.0:

                    if dq > 0.0:  # Local maximum

                        dl = min(dl, min(dq, -dq[0, 0, -1]))

                    else:

                        dl = 0.0

    with computation(PARALLEL), interval(...):

        if irain_f == 0:

            if z_slope_liq == 1:

                # Impose a presumed background horizontal variability that is
                # proportional to the value itself
                dl = max(dl, max(qvmin, h_var * qlz))

            else:

                dl = max(qvmin, h_var * qlz)

            qlz, qrz = autoconv_subgrid_var(
                use_ccn,
                fac_rc,
                t_wfr,
                so3,
                dt_rain,
                qlz,
                qrz,
                tz,
                den,
                ccn,
                c_praut,
                dl,
            )

        rain = rain + r1
        m2_rain = m2_rain + m1_rain

        if is_first == 1:

            m1 = m1 + m1_rain

        else:

            m2_sol = m2_sol + m1_sol
            m1 = m1 + m1_rain + m1_sol


@gtscript.stencil(backend=BACKEND)
def sedimentation(
    graupel: FIELD_FLT,
    ice: FIELD_FLT,
    rain: FIELD_FLT,
    snow: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    den: FIELD_FLT,
    w: FIELD_FLT,
    dz1: FIELD_FLT,
    dp1: FIELD_FLT,
    vtgz: FIELD_FLT,
    vtsz: FIELD_FLT,
    m1_sol: FIELD_FLT,
    do_sedi_w: DTYPE_INT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    d0_vap: DTYPE_FLT,
    lv00: DTYPE_FLT,
    log_10: DTYPE_FLT,
    zs: DTYPE_FLT,
    dts: DTYPE_FLT,
    fac_imlt: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        # Sedimentation of cloud ice, snow, and graupel
        vtgz, vtiz, vtsz = fall_speed(log_10, qgz, qiz, qlz, qsz, tz, den)

        dt5 = 0.5 * dts

        # Define heat capacity and latent heat coefficient
        m1_sol = 0.0

        lhi = li00 + dc_ice * tz
        q_liq = qlz + qrz
        q_sol = qiz + qsz + qgz
        cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
        icpk = lhi / cvm

    # Find significant melting level
    """
    k0 removed to avoid having to introduce a k_idx field
    """
    with computation(BACKWARD):

        with interval(-1, None):

            if tz > tice:
                stop_k = 1
            else:
                stop_k = 0

        with interval(1, -1):

            if stop_k[0, 0, 1] == 0:

                if tz > tice:
                    stop_k = 1
                else:
                    stop_k = 0

            else:

                stop_k = 1

        with interval(0, 1):

            stop_k = 1

    with computation(PARALLEL), interval(...):

        if stop_k == 1:

            # Melting of cloud ice (before fall)
            tc = tz - tice

            if (qiz > qcmin) and (tc > 0.0):

                sink = min(qiz, fac_imlt * tc / icpk)
                tmp = min(sink, dim(ql_mlt, qlz))
                qlz = qlz + tmp
                qrz = qrz + sink - tmp
                qiz = qiz - sink
                q_liq = q_liq + sink
                q_sol = q_sol - sink
                cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz = tz - sink * lhi / cvm
                tc = tz - tice

    with computation(PARALLEL), interval(1, None):

        # Turn off melting when cloud microphysics time step is small
        if dts < 60.0:
            stop_k = 0

        # sjl, turn off melting of falling cloud ice, snow and graupel
        stop_k = 0

    with computation(FORWARD):

        with interval(0, 1):

            ze = zs - dz1

        with interval(1, -1):

            ze = ze[0, 0, -1] - dz1  # dz < 0

        with interval(-1, None):

            ze = ze[0, 0, -1] - dz1  # dz < 0
            zt = ze

    with computation(PARALLEL), interval(...):

        if stop_k == 1:

            # Update capacity heat and latent heat coefficient
            lhi = li00 + dc_ice * tz
            icpk = lhi / cvm

    # Melting of falling cloud ice into rain
    with computation(BACKWARD):

        with interval(-1, None):

            if qiz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        with interval(0, -1):

            if no_fall[0, 0, 1] == 1:

                if qiz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    with computation(FORWARD), interval(1, None):

        if no_fall[0, 0, -1] == 0:
            no_fall = no_fall[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (vi_fac < 1.0e-5) or (no_fall == 1):
            i1 = 0.0

    with computation(PARALLEL):
        with interval(0, 1):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                zt = ze - dt5 * (vtiz[0, 0, 1] + vtiz)
                zt_kbot1 = zs - dts * vtiz
        with interval(1, -1):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):
                zt = ze - dt5 * (vtiz[0, 0, 1] + vtiz)

    with computation(BACKWARD):

        with interval(1, -1):

            if (vi_fac >= 1.0e-5) and (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

        with interval(0, 1):

            if (vi_fac >= 1.0e-5) and (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

            if (vi_fac >= 1.0e-5) and (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    with computation(FORWARD), interval(1, None):

        if (vi_fac >= 1.0e-5) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, -1] - dz_min

    with computation(PARALLEL), interval(...):

        if (vi_fac >= 1.0e-5) and (no_fall == 0):

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    with computation(PARALLEL):
        with interval(0, 1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                dz = ze - zs
        with interval(1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                dz = ze - ze[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):

            dd = dts * vtiz
            qiz = qiz * dp1

    # Sedimentation
    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                qm = qiz / (dz + dd)

        with interval(0, -1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                qm = (qiz[0, 0, 0] + dd[0, 0, 1] * qm[0, 0, 1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                m1_sol = qiz - qm

        with interval(0, -1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                m1_sol = m1_sol[0, 0, 1] + qiz[0, 0, 0] - qm

    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                i1 = m1_sol

        with interval(1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                i1 = i1[0, 0, -1]

    with computation(PARALLEL):
        with interval(0, -1):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                if use_ppm == 0:

                    # Update
                    qiz = qm / dp1

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0]
                        - m1_sol[0, 0, 1] * vtiz[0, 0, 1]
                        + m1_sol * vtiz
                    ) / (dm + m1_sol[0, 0, 1] - m1_sol)
        with interval(-1, None):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                if use_ppm == 0:

                    # Update
                    qiz = qm / dp1

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_sol * vtiz) / (dm - m1_sol)

    # Melting of falling snow into rain
    with computation(BACKWARD):

        with interval(-1, None):

            if qsz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        with interval(0, -1):

            if no_fall[0, 0, 1] == 1:

                if qsz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    with computation(FORWARD), interval(1, None):

        if no_fall[0, 0, -1] == 0:
            no_fall = no_fall[0, 0, -1]

    with computation(PARALLEL), interval(...):

        r1 = 0.0

        if no_fall == 1:
            s1 = 0.0

    with computation(PARALLEL):
        with interval(0, 1):

            if no_fall == 0:

                zt = ze - dt5 * (vtsz[0, 0, 1] + vtsz)
                zt_kbot1 = zs - dts * vtsz

        with interval(1, -1):

            if no_fall == 0:
                zt = ze - dt5 * (vtsz[0, 0, 1] + vtsz)

    with computation(BACKWARD):

        with interval(1, -1):

            if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

        with interval(0, 1):

            if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

            if (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    with computation(FORWARD), interval(1, None):

        if no_fall == 0:
            zt_kbot1 = zt_kbot1[0, 0, -1] - dz_min

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    with computation(PARALLEL):
        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dts * vtsz
            qsz = qsz * dp1

    # Sedimentation
    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qsz / (dz + dd)

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qsz[0, 0, 0] + dd[0, 0, 1] * qm[0, 0, 1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = qsz - qm

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = m1_tf[0, 0, 1] + qsz[0, 0, 0] - qm

    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                s1 = m1_tf

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                s1 = s1[0, 0, -1]

    with computation(PARALLEL):
        with interval(0, -1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qsz = qm / dp1

                m1_sol = m1_sol + m1_tf

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0] - m1_tf[0, 0, 1] * vtsz[0, 0, 1] + m1_tf * vtsz
                    ) / (dm + m1_tf[0, 0, 1] - m1_tf)

        with interval(-1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qsz = qm / dp1

                m1_sol = m1_sol + m1_tf

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtsz) / (dm - m1_tf)

    # Melting of falling graupel into rain
    with computation(BACKWARD):

        with interval(-1, None):

            if qgz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        with interval(0, -1):

            if no_fall[0, 0, 1] == 1:

                if qgz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    with computation(FORWARD), interval(1, None):

        if no_fall[0, 0, -1] == 0:
            no_fall = no_fall[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if no_fall == 1:
            g1 = 0.0

    with computation(PARALLEL):
        with interval(0, 1):

            if no_fall == 0:

                zt = ze - dt5 * (vtgz[0, 0, 1] + vtgz)
                zt_kbot1 = zs - dts * vtgz
        with interval(1, -1):

            if no_fall == 0:
                zt = ze - dt5 * (vtgz[0, 0, 1] + vtgz)

    with computation(BACKWARD):

        with interval(1, -1):

            if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

        with interval(0, 1):

            if (no_fall[0, 0, 1] == 0) and (zt >= zt[0, 0, 1]):
                zt = zt[0, 0, 1] - dz_min

            if (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    with computation(FORWARD), interval(1, None):

        if no_fall == 0:
            zt_kbot1 = zt_kbot1[0, 0, -1] - dz_min

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    with computation(PARALLEL):
        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs
        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, -1]

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dts * vtgz
            qgz = qgz * dp1

    # Sedimentation
    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qgz / (dz + dd)

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qgz[0, 0, 0] + dd[0, 0, 1] * qm[0, 0, 1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = qgz - qm

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = m1_tf[0, 0, 1] + qgz[0, 0, 0] - qm

    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                g1 = m1_tf

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                g1 = g1[0, 0, -1]

    with computation(PARALLEL):
        with interval(0, -1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qgz = qm / dp1

                m1_sol = m1_sol + m1_tf

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0] - m1_tf[0, 0, 1] * vtgz[0, 0, 1] + m1_tf * vtgz
                    ) / (dm + m1_tf[0, 0, 1] - m1_tf)
        with interval(-1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qgz = qm / dp1

                m1_sol = m1_sol + m1_tf

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtgz) / (dm - m1_tf)

    with computation(PARALLEL), interval(...):

        rain = rain + r1  # From melted snow and ice that reached the ground
        snow = snow + s1
        graupel = graupel + g1
        ice = ice + i1

    # Heat transportation during sedimentation
    with computation(PARALLEL):
        with interval(0, -1):

            if do_sedi_heat == 1:

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )
        with interval(-1, None):

            if do_sedi_heat == 1:

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )

                # - Assumption: The ke in the falling condensates is negligible
                #               compared to the potential energy that was
                #               unaccounted for. Local thermal equilibrium is
                #               assumed, and the loss in pe is transformed into
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tmp = cvn + m1_sol * c_ice
                tz = tz + m1_sol * dgz / tmp

    # Implicit algorithm
    with computation(BACKWARD), interval(0, -1):

        if do_sedi_heat == 1:

            tz[0, 0, 0] = (
                (cvn + c_ice * (m1_sol - m1_sol[0, 0, 1])) * tz[0, 0, 0]
                + m1_sol[0, 0, 1] * c_ice * tz[0, 0, 1]
                + dgz * (m1_sol[0, 0, 1] + m1_sol)
            ) / (cvn + c_ice * m1_sol)


@gtscript.stencil(backend=BACKEND)
def icloud(
    h_var: FIELD_FLT,
    rh_adj: FIELD_FLT,
    rh_rain: FIELD_FLT,
    qaz: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    den: FIELD_FLT,
    denfac: FIELD_FLT,
    p1: FIELD_FLT,
    vtgz: FIELD_FLT,
    vtrz: FIELD_FLT,
    vtsz: FIELD_FLT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    d0_vap: DTYPE_FLT,
    lv00: DTYPE_FLT,
    cracs: DTYPE_FLT,
    csacr: DTYPE_FLT,
    cgacr: DTYPE_FLT,
    cgacs: DTYPE_FLT,
    acco_00: DTYPE_FLT,
    acco_01: DTYPE_FLT,
    acco_02: DTYPE_FLT,
    acco_03: DTYPE_FLT,
    acco_10: DTYPE_FLT,
    acco_11: DTYPE_FLT,
    acco_12: DTYPE_FLT,
    acco_13: DTYPE_FLT,
    acco_20: DTYPE_FLT,
    acco_21: DTYPE_FLT,
    acco_22: DTYPE_FLT,
    acco_23: DTYPE_FLT,
    csacw: DTYPE_FLT,
    csaci: DTYPE_FLT,
    cgacw: DTYPE_FLT,
    cgaci: DTYPE_FLT,
    cracw: DTYPE_FLT,
    cssub_0: DTYPE_FLT,
    cssub_1: DTYPE_FLT,
    cssub_2: DTYPE_FLT,
    cssub_3: DTYPE_FLT,
    cssub_4: DTYPE_FLT,
    cgfr_0: DTYPE_FLT,
    cgfr_1: DTYPE_FLT,
    csmlt_0: DTYPE_FLT,
    csmlt_1: DTYPE_FLT,
    csmlt_2: DTYPE_FLT,
    csmlt_3: DTYPE_FLT,
    csmlt_4: DTYPE_FLT,
    cgmlt_0: DTYPE_FLT,
    cgmlt_1: DTYPE_FLT,
    cgmlt_2: DTYPE_FLT,
    cgmlt_3: DTYPE_FLT,
    cgmlt_4: DTYPE_FLT,
    ces0: DTYPE_FLT,
    tice0: DTYPE_FLT,
    t_wfr: DTYPE_FLT,
    dts: DTYPE_FLT,
    rdts: DTYPE_FLT,
    fac_i2s: DTYPE_FLT,
    fac_g2v: DTYPE_FLT,
    fac_v2g: DTYPE_FLT,
    fac_imlt: DTYPE_FLT,
    fac_l2v: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        # Ice-phase microphysics

        # Define heat capacity and latent heat coefficient
        lhi = li00 + dc_ice * tz
        q_liq = qlz + qrz
        q_sol = qiz + qsz + qgz
        cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
        icpk = lhi / cvm

        # - Sources of cloud ice: pihom, cold rain, and the sat_adj
        # - Sources of snow: cold rain, auto conversion + accretion (from cloud ice)
        # - sat_adj (deposition; requires pre-existing snow); initial snow comes from autoconversion

        t_wfr_tmp = t_wfr
        if (tz > tice) and (qiz > qcmin):

            # pimlt: instant melting of cloud ice
            melt = min(qiz, fac_imlt * (tz - tice) / icpk)
            tmp = min(melt, dim(ql_mlt, qlz))  # Maximum ql amount
            qlz = qlz + tmp
            qrz = qrz + melt - tmp
            qiz = qiz - melt
            q_liq = q_liq + melt
            q_sol = q_sol - melt
            cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            tz = tz - melt * lhi / cvm

        elif (tz < t_wfr) and (qlz > qcmin):

            # - pihom: homogeneous freezing of cloud water into cloud ice
            # - This is the 1st occurance of liquid water freezing in the split mp process

            dtmp = t_wfr_tmp - tz
            factor = min(1.0, dtmp / dt_fr)
            sink = min(qlz * factor, dtmp / icpk)
            qi_crt = qi_gen * min(qi_lim, 0.1 * (tice - tz)) / den
            tmp = min(sink, dim(qi_crt, qiz))
            qlz = qlz - sink
            qsz = qsz + sink - tmp
            qiz = qiz + tmp
            q_liq = q_liq - sink
            q_sol = q_sol + sink
            cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            tz = tz + sink * lhi / cvm

    # Vertical subgrid variability
    with computation(BACKWARD):

        with interval(-1, None):

            if z_slope_ice == 1:
                di = 0.0

        with interval(0, -1):

            if z_slope_ice == 1:
                dq = 0.5 * (qiz[0, 0, 0] - qiz[0, 0, 1])

    with computation(PARALLEL):
        with interval(0, 1):

            if z_slope_ice == 1:
                di = 0.0

        with interval(1, -1):

            if z_slope_ice == 1:

                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                di = 0.5 * min(abs(dq + dq[0, 0, -1]), 0.5 * qiz[0, 0, 0])

                if dq * dq[0, 0, -1] <= 0.0:

                    if dq > 0.0:  # Local maximum

                        di = min(di, min(dq, -dq[0, 0, -1]))

                    else:

                        di = 0.0

    with computation(PARALLEL), interval(...):

        if z_slope_ice == 1:

            # Impose a presumed background horizontal variability that is
            # proportional to the value itself
            di = max(di, max(qvmin, h_var * qiz))

        else:

            di = max(qvmin, h_var * qiz)

        qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz = icloud_main(
            c_air,
            c_vap,
            d0_vap,
            lv00,
            cracs,
            csacr,
            cgacr,
            cgacs,
            acco_00,
            acco_01,
            acco_02,
            acco_03,
            acco_10,
            acco_11,
            acco_12,
            acco_13,
            acco_20,
            acco_21,
            acco_22,
            acco_23,
            csacw,
            csaci,
            cgacw,
            cgaci,
            cssub_0,
            cssub_1,
            cssub_2,
            cssub_3,
            cssub_4,
            cgfr_0,
            cgfr_1,
            csmlt_0,
            csmlt_1,
            csmlt_2,
            csmlt_3,
            csmlt_4,
            cgmlt_0,
            cgmlt_1,
            cgmlt_2,
            cgmlt_3,
            cgmlt_4,
            ces0,
            tice0,
            t_wfr,
            dts,
            rdts,
            fac_i2s,
            fac_g2v,
            fac_v2g,
            fac_l2v,
            h_var,
            rh_adj,
            rh_rain,
            qaz,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            denfac,
            vtgz,
            vtrz,
            vtsz,
            p1,
            di,
            q_liq,
            q_sol,
            cvm,
        )


@gtscript.stencil(backend=BACKEND)
def fields_update(
    graupel: FIELD_FLT,
    ice: FIELD_FLT,
    rain: FIELD_FLT,
    snow: FIELD_FLT,
    qaz: FIELD_FLT,
    qgz: FIELD_FLT,
    qiz: FIELD_FLT,
    qlz: FIELD_FLT,
    qrz: FIELD_FLT,
    qsz: FIELD_FLT,
    qvz: FIELD_FLT,
    tz: FIELD_FLT,
    udt: FIELD_FLT,
    vdt: FIELD_FLT,
    qa_dt: FIELD_FLT,
    qg_dt: FIELD_FLT,
    qi_dt: FIELD_FLT,
    ql_dt: FIELD_FLT,
    qr_dt: FIELD_FLT,
    qs_dt: FIELD_FLT,
    qv_dt: FIELD_FLT,
    pt_dt: FIELD_FLT,
    qa0: FIELD_FLT,
    qg0: FIELD_FLT,
    qi0: FIELD_FLT,
    ql0: FIELD_FLT,
    qr0: FIELD_FLT,
    qs0: FIELD_FLT,
    qv0: FIELD_FLT,
    t0: FIELD_FLT,
    dp0: FIELD_FLT,
    u0: FIELD_FLT,
    v0: FIELD_FLT,
    dp1: FIELD_FLT,
    u1: FIELD_FLT,
    v1: FIELD_FLT,
    m1: FIELD_FLT,
    m2_rain: FIELD_FLT,
    m2_sol: FIELD_FLT,
    ntimes: DTYPE_INT,
    c_air: DTYPE_FLT,
    c_vap: DTYPE_FLT,
    rdt: DTYPE_FLT,
):

    with computation(PARALLEL), interval(...):

        # Convert units from Pa*kg/kg to kg/m^2/s
        m2_rain = m2_rain * rdt * rgrav
        m2_sol = m2_sol * rdt * rgrav

    # Momentum transportation during sedimentation (dp1 is dry mass; dp0
    # is the old moist total mass)
    with computation(BACKWARD), interval(0, -1):

        if sedi_transport == 1:

            u1[0, 0, 0] = (dp0[0, 0, 0] * u1[0, 0, 0] + m1[0, 0, 1] * u1[0, 0, 1]) / (
                dp0[0, 0, 0] + m1[0, 0, 1]
            )
            v1[0, 0, 0] = (dp0[0, 0, 0] * v1[0, 0, 0] + m1[0, 0, 1] * v1[0, 0, 1]) / (
                dp0[0, 0, 0] + m1[0, 0, 1]
            )

    with computation(PARALLEL), interval(0, -1):

        if sedi_transport == 1:

            udt = udt + (u1 - u0) * rdt
            vdt = vdt + (v1 - v0) * rdt

    with computation(PARALLEL), interval(...):

        # Update moist air mass (actually hydrostatic pressure) and convert
        # to dry mixing ratios
        omq = dp1 / dp0
        qv_dt = qv_dt + rdt * (qvz - qv0) * omq
        ql_dt = ql_dt + rdt * (qlz - ql0) * omq
        qr_dt = qr_dt + rdt * (qrz - qr0) * omq
        qi_dt = qi_dt + rdt * (qiz - qi0) * omq
        qs_dt = qs_dt + rdt * (qsz - qs0) * omq
        qg_dt = qg_dt + rdt * (qgz - qg0) * omq

        cvm = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice

        pt_dt = pt_dt + rdt * (tz - t0) * cvm / cp_air

        # Update cloud fraction tendency
        if do_qa == 1:

            qa_dt = 0.0

        else:

            qa_dt = qa_dt + rdt * (qaz / ntimes - qa0)

        """
        LEFT OUT FOR NOW
        # No clouds allowed above ktop
        if k_s < ktop:
            qa_dt[:, :, k_s:ktop+1] = 0.
        """

        # Convert to mm / day
        convt = 86400.0 * rdt * rgrav

        rain = rain * convt
        snow = snow * convt
        ice = ice * convt
        graupel = graupel * convt


c_air = None
c_vap = None
d0_vap = None  # The same as dc_vap, except that cp_vap can be cp_vap or cv_vap
lv00 = None  # The same as lv0, except that cp_vap can be cp_vap or cv_vap
fac_rc = None
cracs = None
csacr = None
cgacr = None
cgacs = None
acco = None
csacw = None
csaci = None
cgacw = None
cgaci = None
cracw = None
cssub = None
crevp = None
cgfr = None
csmlt = None
cgmlt = None
ces0 = None
log_10 = None
tice0 = None
t_wfr = None

do_sedi_w = 1  # Transport of vertical motion in sedimentation
do_setup = True  # Setup constants and parameters
p_nonhydro = 0  # Perform hydrosatic adjustment on air density
use_ccn = 1  # Must be true when prog_ccn is false


def setupm():

    # Global variables
    global fac_rc
    global cracs
    global csacr
    global cgacr
    global cgacs
    global acco
    global csacw
    global csaci
    global cgacw
    global cgaci
    global cracw
    global cssub
    global crevp
    global cgfr
    global csmlt
    global cgmlt
    global ces0

    gam263 = 1.456943
    gam275 = 1.608355
    gam290 = 1.827363
    gam325 = 2.54925
    gam350 = 3.323363
    gam380 = 4.694155

    # Intercept parameters
    rnzs = 3.0e6
    rnzr = 8.0e6
    rnzg = 4.0e6

    # Density parameters
    acc = np.array([5.0, 2.0, 0.5])

    pie = 4.0 * mt.atan(1.0)

    # S. Klein's formular (eq 16) from am2
    fac_rc = (4.0 / 3.0) * pie * rhor * rthresh ** 3

    vdifu = 2.11e-5
    tcond = 2.36e-2

    visk = 1.259e-5
    hlts = 2.8336e6
    hltc = 2.5e6
    hltf = 3.336e5

    ch2o = 4.1855e3

    pisq = pie * pie
    scm3 = (visk / vdifu) ** (1.0 / 3.0)

    cracs = pisq * rnzr * rnzs * rhos
    csacr = pisq * rnzr * rnzs * rhor
    cgacr = pisq * rnzr * rnzg * rhor
    cgacs = pisq * rnzg * rnzs * rhos
    cgacs = cgacs * c_pgacs

    act = np.empty(8)
    act[0] = pie * rnzs * rhos
    act[1] = pie * rnzr * rhor
    act[5] = pie * rnzg * rhog
    act[2] = act[1]
    act[3] = act[0]
    act[4] = act[1]
    act[6] = act[0]
    act[7] = act[5]

    acco = np.empty((3, 4))
    for i in range(3):
        for k in range(4):
            acco[i, k] = acc[i] / (
                act[2 * k] ** ((6 - i) * 0.25) * act[2 * k + 1] ** ((i + 1) * 0.25)
            )

    gcon = 40.74 * mt.sqrt(sfcrho)

    # Decreasing csacw to reduce cloud water --- > snow
    csacw = pie * rnzs * clin * gam325 / (4.0 * act[0] ** 0.8125)

    craci = pie * rnzr * alin * gam380 / (4.0 * act[1] ** 0.95)
    csaci = csacw * c_psaci

    cgacw = pie * rnzg * gam350 * gcon / (4.0 * act[5] ** 0.875)

    cgaci = cgacw * 0.05

    cracw = craci
    cracw = c_cracw * cracw

    # Subl and revap: five constants for three separate processes
    cssub = np.empty(5)
    cssub[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzs
    cssub[1] = 0.78 / mt.sqrt(act[0])
    cssub[2] = 0.31 * scm3 * gam263 * mt.sqrt(clin / visk) / act[0] ** 0.65625
    cssub[3] = tcond * rvgas
    cssub[4] = (hlts ** 2) * vdifu

    cgsub = np.empty(5)
    cgsub[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzg
    cgsub[1] = 0.78 / mt.sqrt(act[5])
    cgsub[2] = 0.31 * scm3 * gam275 * mt.sqrt(gcon / visk) / act[5] ** 0.6875
    cgsub[3] = cssub[3]
    cgsub[4] = cssub[4]

    crevp = np.empty(5)
    crevp[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzr
    crevp[1] = 0.78 / mt.sqrt(act[1])
    crevp[2] = 0.31 * scm3 * gam290 * mt.sqrt(alin / visk) / act[1] ** 0.725
    crevp[3] = cssub[3]
    crevp[4] = hltc ** 2 * vdifu

    cgfr = np.empty(2)
    cgfr[0] = 20.0e2 * pisq * rnzr * rhor / act[1] ** 1.75
    cgfr[1] = 0.66

    # smlt: five constants (lin et al. 1983)
    csmlt = np.empty(5)
    csmlt[0] = 2.0 * pie * tcond * rnzs / hltf
    csmlt[1] = 2.0 * pie * vdifu * rnzs * hltc / hltf
    csmlt[2] = cssub[1]
    csmlt[3] = cssub[2]
    csmlt[4] = ch2o / hltf

    # gmlt: five constants
    cgmlt = np.empty(5)
    cgmlt[0] = 2.0 * pie * tcond * rnzg / hltf
    cgmlt[1] = 2.0 * pie * vdifu * rnzg * hltc / hltf
    cgmlt[2] = cgsub[1]
    cgmlt[3] = cgsub[2]
    cgmlt[4] = ch2o / hltf

    es0 = 6.107799961e2  # ~6.1 mb
    ces0 = eps * es0


def gfdl_cloud_microphys_init():

    # Global variables
    global log_10
    global tice0
    global t_wfr

    global do_setup

    if do_setup:
        setupm()
        do_setup = False

    log_10 = mt.log(10.0)
    tice0 = tice - 0.01
    t_wfr = tice - 40.0


def run(input_data):
    gfdl_cloud_microphys_init()
    input_data = scale_dataset(input_data, (1.0, 1))
    if BACKEND == "gtx86" or BACKEND == "gtcuda" or BACKEND == "numpy":
        input_data = numpy_dict_to_gt4py_dict(input_data)
    hydrostatic = False
    phys_hydrostatic = True
    kks = 0
    ktop = 0
    # Scalar input values (-1 for indices, since ported from Fortran)
    kke = input_data["kke"] - 1  # End of vertical dimension
    kbot = input_data["kbot"] - 1  # Bottom of vertical compute domain
    dt_in = input_data["dt_in"]  # Physics time step

    # 2D input arrays
    area = input_data["area"]  # Cell area
    land = input_data["land"]  # Land fraction
    rain = input_data["rain"]
    snow = input_data["snow"]
    ice = input_data["ice"]
    graupel = input_data["graupel"]

    # 3D input arrays
    dz = input_data["dz"]
    delp = input_data["delp"]
    uin = input_data["uin"]
    vin = input_data["vin"]
    qv = input_data["qv"]
    ql = input_data["ql"]
    qr = input_data["qr"]
    qi = input_data["qi"]
    qs = input_data["qs"]
    qg = input_data["qg"]
    qa = input_data["qa"]
    qn = input_data["qn"]
    p = input_data["p"]
    pt = input_data["pt"]
    qv_dt = input_data["qv_dt"]
    ql_dt = input_data["ql_dt"]
    qr_dt = input_data["qr_dt"]
    qi_dt = input_data["qi_dt"]
    qs_dt = input_data["qs_dt"]
    qg_dt = input_data["qg_dt"]
    qa_dt = input_data["qa_dt"]
    pt_dt = input_data["pt_dt"]
    udt = input_data["udt"]
    vdt = input_data["vdt"]
    w = input_data["w"]
    refl_10cm = input_data["refl_10cm"]

    # Common 3D shape of all gt4py storages
    shape = qi.shape

    # 2D local arrays
    h_var = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    rh_adj = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    rh_rain = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)

    # 3D local arrays
    qaz = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qgz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qiz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qlz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qrz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qsz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qvz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    den = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    denfac = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    tz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qa0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qg0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qi0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    ql0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qr0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qs0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    qv0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    t0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    dp0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    den0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    dz0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    u0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    v0 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    dz1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    dp1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    p1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    u1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    v1 = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    m1 = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    vtgz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    vtrz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    vtsz = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    ccn = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    c_praut = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    m1_sol = gt.storage.empty(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    m2_rain = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)
    m2_sol = gt.storage.zeros(BACKEND, DEFAULT_ORIGIN, shape, dtype=DTYPE_FLT)

    # Global variables
    global c_air
    global c_vap
    global d0_vap
    global lv00

    global do_sedi_w
    global p_nonhydro
    global use_ccn

    # Define start and end indices of the vertical dimensions
    k_s = kks
    k_e = kke - kks + 1

    # Define heat capacity of dry air and water vapor based on
    # hydrostatical property
    if phys_hydrostatic or hydrostatic:

        c_air = cp_air
        c_vap = cp_vap
        p_nonhydro = 0

    else:

        c_air = cv_air
        c_vap = cv_vap
        p_nonhydro = 1

    d0_vap = c_vap - c_liq
    lv00 = hlv0 - d0_vap * t_ice

    if hydrostatic:
        do_sedi_w = 0

    # Define cloud microphysics sub time step
    mpdt = np.minimum(dt_in, mp_time)
    rdt = 1.0 / dt_in
    ntimes = DTYPE_INT(round(dt_in / mpdt))

    # Small time step
    dts = dt_in / ntimes

    dt_rain = dts * 0.5

    # Calculate cloud condensation nuclei (ccn) based on klein eq. 15
    cpaut = c_paut * 0.104 * grav / 1.717e-5

    # Set use_ccn to false if prog_ccn is true
    if prog_ccn == 1:
        use_ccn = 0
    exec_info = {}
    ### Major cloud microphysics ###
    fields_init(
        land,
        area,
        h_var,
        rh_adj,
        rh_rain,
        graupel,
        ice,
        rain,
        snow,
        qa,
        qg,
        qi,
        ql,
        qn,
        qr,
        qs,
        qv,
        pt,
        delp,
        dz,
        qgz,
        qiz,
        qlz,
        qrz,
        qsz,
        qvz,
        tz,
        qi_dt,
        qs_dt,
        uin,
        vin,
        qa0,
        qg0,
        qi0,
        ql0,
        qr0,
        qs0,
        qv0,
        t0,
        dp0,
        den0,
        dz0,
        u0,
        v0,
        dp1,
        p1,
        u1,
        v1,
        ccn,
        c_praut,
        DTYPE_INT(use_ccn),
        c_air,
        c_vap,
        d0_vap,
        lv00,
        dt_in,
        rdt,
        cpaut,
        exec_info=exec_info,
    )
    so3 = 7.0 / 3.0

    zs = 0.0

    rdts = 1.0 / dts

    if fast_sat_adj:
        dt_evap = 0.5 * dts
    else:
        dt_evap = dts

    # Define conversion scalar / factor
    fac_i2s = 1.0 - mt.exp(-dts / tau_i2s)
    fac_g2v = 1.0 - mt.exp(-dts / tau_g2v)
    fac_v2g = 1.0 - mt.exp(-dts / tau_v2g)
    fac_imlt = 1.0 - mt.exp(-0.5 * dts / tau_imlt)
    fac_l2v = 1.0 - mt.exp(-dt_evap / tau_l2v)

    for n in range(ntimes):

        exec_info = {}

        # Time-split warm rain processes: 1st pass
        warm_rain(
            h_var,
            rain,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            denfac,
            w,
            t0,
            den0,
            dz0,
            dz1,
            dp1,
            m1,
            vtrz,
            ccn,
            c_praut,
            m1_sol,
            m2_rain,
            m2_sol,
            DTYPE_INT(1),
            DTYPE_INT(do_sedi_w),
            DTYPE_INT(p_nonhydro),
            DTYPE_INT(use_ccn),
            c_air,
            c_vap,
            d0_vap,
            lv00,
            fac_rc,
            cracw,
            crevp[0],
            crevp[1],
            crevp[2],
            crevp[3],
            crevp[4],
            t_wfr,
            so3,
            dt_rain,
            zs,
            exec_info=exec_info,
        )
        exec_info = {}

        # Sedimentation of cloud ice, snow, and graupel
        sedimentation(
            graupel,
            ice,
            rain,
            snow,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            w,
            dz1,
            dp1,
            vtgz,
            vtsz,
            m1_sol,
            DTYPE_INT(do_sedi_w),
            c_air,
            c_vap,
            d0_vap,
            lv00,
            log_10,
            zs,
            dts,
            fac_imlt,
            exec_info=exec_info,
        )
        exec_info = {}

        # Time-split warm rain processes: 2nd pass
        warm_rain(
            h_var,
            rain,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            denfac,
            w,
            t0,
            den0,
            dz0,
            dz1,
            dp1,
            m1,
            vtrz,
            ccn,
            c_praut,
            m1_sol,
            m2_rain,
            m2_sol,
            DTYPE_INT(0),
            DTYPE_INT(do_sedi_w),
            DTYPE_INT(p_nonhydro),
            DTYPE_INT(use_ccn),
            c_air,
            c_vap,
            d0_vap,
            lv00,
            fac_rc,
            cracw,
            crevp[0],
            crevp[1],
            crevp[2],
            crevp[3],
            crevp[4],
            t_wfr,
            so3,
            dt_rain,
            zs,
            exec_info=exec_info,
        )

        exec_info = {}

        # Ice-phase microphysics
        icloud(
            h_var,
            rh_adj,
            rh_rain,
            qaz,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            denfac,
            p1,
            vtgz,
            vtrz,
            vtsz,
            c_air,
            c_vap,
            d0_vap,
            lv00,
            cracs,
            csacr,
            cgacr,
            cgacs,
            acco[0, 0],
            acco[0, 1],
            acco[0, 2],
            acco[0, 3],
            acco[1, 0],
            acco[1, 1],
            acco[1, 2],
            acco[1, 3],
            acco[2, 0],
            acco[2, 1],
            acco[2, 2],
            acco[2, 3],
            csacw,
            csaci,
            cgacw,
            cgaci,
            cracw,
            cssub[0],
            cssub[1],
            cssub[2],
            cssub[3],
            cssub[4],
            cgfr[0],
            cgfr[1],
            csmlt[0],
            csmlt[1],
            csmlt[2],
            csmlt[3],
            csmlt[4],
            cgmlt[0],
            cgmlt[1],
            cgmlt[2],
            cgmlt[3],
            cgmlt[4],
            ces0,
            tice0,
            t_wfr,
            dts,
            rdts,
            fac_i2s,
            fac_g2v,
            fac_v2g,
            fac_imlt,
            fac_l2v,
            exec_info=exec_info,
        )
        exec_info = {}

    exec_info = {}
    fields_update(
        graupel,
        ice,
        rain,
        snow,
        qaz,
        qgz,
        qiz,
        qlz,
        qrz,
        qsz,
        qvz,
        tz,
        udt,
        vdt,
        qa_dt,
        qg_dt,
        qi_dt,
        ql_dt,
        qr_dt,
        qs_dt,
        qv_dt,
        pt_dt,
        qa0,
        qg0,
        qi0,
        ql0,
        qr0,
        qs0,
        qv0,
        t0,
        dp0,
        u0,
        v0,
        dp1,
        u1,
        v1,
        m1,
        m2_rain,
        m2_sol,
        ntimes,
        c_air,
        c_vap,
        rdt,
        exec_info=exec_info,
    )
    """
    NOTE: Radar part missing (never executed since lradar is false)
    """

    output = view_gt4py_storage(
        {
            "qi": qi[:, :, :],
            "qs": qs[:, :, :],
            "qv_dt": qv_dt[:, :, :],
            "ql_dt": ql_dt[:, :, :],
            "qr_dt": qr_dt[:, :, :],
            "qi_dt": qi_dt[:, :, :],
            "qs_dt": qs_dt[:, :, :],
            "qg_dt": qg_dt[:, :, :],
            "qa_dt": qa_dt[:, :, :],
            "pt_dt": pt_dt[:, :, :],
            "w": w[:, :, :],
            "udt": udt[:, :, :],
            "vdt": vdt[:, :, :],
            "rain": rain[:, :, 0],
            "snow": snow[:, :, 0],
            "ice": ice[:, :, 0],
            "graupel": graupel[:, :, 0],
            "refl_10cm": refl_10cm[:, :, :],
        }
    )
    return output
