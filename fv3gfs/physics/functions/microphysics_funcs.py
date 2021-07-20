from fv3gfsphysics.utils.global_constants import *
from gt4py import gtscript


@gtscript.function
def dim(x, y):

    diff = x - y

    return diff if diff > 0.0 else 0.0


# Compute the saturated specific humidity
@gtscript.function
def wqs1(ta, den):

    return (
        e00
        * exp((dc_vap * log(ta / t_ice) + lv0 * (ta - t_ice) / (ta * t_ice)) / rvgas)
    ) / (rvgas * ta * den)


# Compute saturated specific humidity and its gradient
@gtscript.function
def wqs2(ta, den):

    tmp = wqs1(ta, den)

    return tmp, tmp * (dc_vap + lv0 / ta) / (rvgas * ta)


# Compute the saturated specific humidity
@gtscript.function
def iqs1(ta, den):

    if ta < t_ice:

        # Over ice between -160 degrees Celsius and 0 degrees Celsius
        if ta >= t_ice - 160.0:

            tmp = (
                e00
                * exp(
                    (d2ice * log(ta / t_ice) + li2 * (ta - t_ice) / (ta * t_ice))
                    / rvgas
                )
            ) / (rvgas * ta * den)

        else:

            tmp = (
                e00
                * exp(
                    (
                        d2ice * log(1.0 - 160.0 / t_ice)
                        - li2 * 160.0 / ((t_ice - 160.0) * t_ice)
                    )
                    / rvgas
                )
            ) / (rvgas * (t_ice - 160.0) * den)
    else:

        # Over water between 0 degrees Celsius and 102 degrees Celsius
        if ta <= t_ice + 102.0:

            tmp = wqs1(ta, den)

        else:

            tmp = wqs1(t_ice + 102.0, den)

    return tmp


# Compute the gradient of saturated specific humidity
@gtscript.function
def iqs2(ta, den):

    tmp = iqs1(ta, den)

    if ta < t_ice:

        # Over ice between -160 degrees Celsius and 0 degrees Celsius
        if ta >= t_ice - 160.0:

            dtmp = tmp * (d2ice + li2 / ta) / (rvgas * ta)

        else:

            dtmp = tmp * (d2ice + li2 / (t_ice - 160.0)) / (rvgas * (t_ice - 160.0))

    else:

        # Over water between 0 degrees Celsius and 102 degrees Celsius
        if ta <= t_ice + 102.0:

            dtmp = tmp * (dc_vap + lv0 / ta) / (rvgas * ta)

        else:

            dtmp = tmp * (dc_vap + lv0 / (t_ice + 102.0)) / (rvgas * (t_ice + 102.0))

    return tmp, dtmp


# Accretion function
@gtscript.function
def acr3d(v1, v2, q1, q2, c, cac_ik, cac_i1k, cac_i2k, rho):

    t1 = sqrt(q1 * rho)
    s1 = sqrt(q2 * rho)
    s2 = sqrt(s1)

    return (
        c
        * abs(v1 - v2)
        * q1
        * s2
        * (cac_ik * t1 + cac_i1k * sqrt(t1) * s2 + cac_i2k * s1)
    )


# Melting of snow function (psacw and psacr must be calc before smlt is
# called)
@gtscript.function
def smlt(tc, dqs, qsrho, psacw, psacr, c_0, c_1, c_2, c_3, c_4, rho, rhofac):

    return (c_0 * tc / rho - c_1 * dqs) * (
        c_2 * sqrt(qsrho) + c_3 * qsrho ** 0.65625 * sqrt(rhofac)
    ) + c_4 * tc * (psacw + psacr)


# Melting of graupel function (pgacw and pgacr must be calc before gmlt
# is called)
@gtscript.function
def gmlt(tc, dqs, qgrho, pgacw, pgacr, c_0, c_1, c_2, c_3, c_4, rho):

    return (c_0 * tc / rho - c_1 * dqs) * (
        c_2 * sqrt(qgrho) + c_3 * qgrho ** 0.6875 / rho ** 0.25
    ) + c_4 * tc * (pgacw + pgacr)


# Evaporation of rain
@gtscript.function
def revap_racc(
    dt,
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
    qg,
    qi,
    ql,
    qr,
    qs,
    qv,
    tz,
    den,
    denfac,
):

    # Evaporation and accretion of rain for the first 1/2 time step
    if (tz > t_wfr) and (qr > qrmin):

        # Define heat capacity and latent heat coefficient
        lhl = lv00 + d0_vap * tz
        q_liq = ql + qr
        q_sol = qi + qs + qg
        cvm = c_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice
        lcpk = lhl / cvm

        tin = tz - lcpk * ql  # Presence of clouds suppresses the rain evap
        qpz = qv + ql

        qsat, dqsdt = wqs2(tin, den)

        dqh = max(ql, h_var * max(qpz, qcmin))
        dqh = min(dqh, 0.2 * qpz)  # New limiter
        dqv = qsat - qv  # Use this to prevent super-sat the gird box
        q_minus = qpz - dqh
        q_plus = qpz + dqh

        # qsat must be > q_minus to activate evaporation
        # qsat must be < q_plus to activate accretion
        # Rain evaporation
        if (dqv > qvmin) and (qsat > q_minus):

            if qsat > q_plus:

                dq = qsat - qpz

            else:

                # q_minus < qsat < q_plus
                # dq == dqh if qsat == q_minus
                dq = 0.25 * (q_minus - qsat) ** 2 / dqh

            qden = qr * den
            t2 = tin * tin
            evap = (
                crevp_0
                * t2
                * dq
                * (crevp_1 * sqrt(qden) + crevp_2 * exp(0.725 * log(qden)))
                / (crevp_3 * t2 + crevp_4 * qsat * den)
            )
            evap = min(qr, min(dt * evap, dqv / (1.0 + lcpk * dqsdt)))

            # Alternative minimum evap in dry environmental air
            qr = qr - evap
            qv = qv + evap
            q_liq = q_liq - evap
            cvm = c_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice
            tz = tz - evap * lhl / cvm

        # Accretion: pracc
        if (qr > qrmin) and (ql > 1.0e-6) and (qsat < q_minus):

            sink = dt * denfac * cracw * exp(0.95 * log(qr * den))
            sink = sink / (1.0 + sink) * ql
            ql = ql - sink
            qr = qr + sink

    return qg, qi, ql, qr, qs, qv, tz


# Calculate the vertical fall speed
@gtscript.function
def fall_speed(log_10, qg, qi, ql, qs, tk, den):

    # Marshall-Palmer formula

    # Try the local air density -- for global model; the true value
    # could be much smaller than sfcrho over high mountains
    rhof = sqrt(min(10.0, sfcrho / den))

    # Ice
    if const_vi == 1:

        vti = vi_fac

    else:

        # Use deng and mace (2008, grl), which gives smaller fall speed
        # than hd90 formula
        vi0 = 0.01 * vi_fac

        if qi < thi:

            vti = vf_min

        else:

            tc = tk - tice
            """
            THE LOG10 HAD TO BE TRANSFORMED DUE TO THE LOG10 FUNCTION NOT BEING IMPLEMENTED (?)
            """
            # ~ vti = (3. + log10(qi * den)) * (tc * (aa * tc + bb) + cc) + dd_fs * tc + ee
            vti = (
                (3.0 + log(qi * den) / log_10) * (tc * (aa * tc + bb) + cc)
                + dd_fs * tc
                + ee
            )
            vti = vi0 * exp(log_10 * vti) * 0.8
            vti = min(vi_max, max(vf_min, vti))

    # Snow
    if const_vs == 1:

        vts = vs_fac

    else:

        if qs < ths:

            vts = vf_min

        else:

            vts = vs_fac * vcons * rhof * exp(0.0625 * log(qs * den / norms))
            vts = min(vs_max, max(vf_min, vts))

    # Graupel
    if const_vg == 1:

        vtg = vg_fac

    else:

        if qg < thg:

            vtg = vf_min

        else:

            vtg = vg_fac * vcong * rhof * sqrt(sqrt(sqrt(qg * den / normg)))
            vtg = min(vg_max, max(vf_min, vtg))

    return vtg, vti, vts


@gtscript.function
def compute_rain_fspeed(no_fall, qrz, den):

    if no_fall == 1:

        vtrz = vf_min
        r1 = 0.0

    else:

        # Fall speed of rain
        if const_vr == 1:

            vtrz = vr_fac

        else:

            qden = qrz * den

            if qrz < thr:

                vtrz = vr_min

            else:

                vtrz = (
                    vr_fac
                    * vconr
                    * sqrt(min(10.0, sfcrho / den))
                    * exp(0.2 * log(qden / normr))
                )
                vtrz = min(vr_max, max(vr_min, vtrz))

    return vtrz, r1


@gtscript.function
def autoconv_no_subgrid_var(
    use_ccn, fac_rc, t_wfr, so3, dt_rain, qlz, qrz, tz, den, ccn, c_praut
):

    # No subgrid variability
    qc0 = fac_rc * ccn

    if tz > t_wfr:

        if use_ccn == 1:

            # ccn is formulted as ccn = ccn_surface * (den / den_surface)
            qc = qc0

        else:

            qc = qc0 / den

        dq = qlz - qc

        if dq > 0.0:

            sink = min(dq, dt_rain * c_praut * den * exp(so3 * log(qlz)))
            qlz = qlz - sink
            qrz = qrz + sink

    return qlz, qrz


@gtscript.function
def autoconv_subgrid_var(
    use_ccn, fac_rc, t_wfr, so3, dt_rain, qlz, qrz, tz, den, ccn, c_praut, dl
):

    qc0 = fac_rc * ccn

    if tz > t_wfr + dt_fr:

        dl = min(max(1.0e-6, dl), 0.5 * qlz)

        # As in klein's gfdl am2 stratiform scheme (with subgrid variations)
        if use_ccn == 1:

            # ccn is formulted as ccn = ccn_surface * (den / den_surface)
            qc = qc0

        else:

            qc = qc0 / den

        dq = 0.5 * (qlz + dl - qc)

        # dq = dl if qc == q_minus = ql - dl
        # dq = 0 if qc == q_plus = ql + dl
        if dq > 0.0:  # q_plus > qc

            # Revised continuous form: linearly decays (with subgrid dl) to zero at qc == ql + dl
            sink = min(1.0, dq / dl) * dt_rain * c_praut * den * exp(so3 * log(qlz))
            qlz = qlz - sink
            qrz = qrz + sink

    return qlz, qrz


@gtscript.function
def subgrid_z_proc(
    c_air,
    c_vap,
    d0_vap,
    lv00,
    cssub_0,
    cssub_1,
    cssub_2,
    cssub_3,
    cssub_4,
    t_wfr,
    dts,
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
    p1,
):

    # Subgrid cloud microphysics
    # Define heat capacity and latent heat coefficient
    lhl = lv00 + d0_vap * tz
    lhi = li00 + dc_ice * tz
    q_liq = qlz + qrz
    q_sol = qiz + qsz + qgz
    cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
    lcpk = lhl / cvm
    icpk = lhi / cvm
    tcpk = lcpk + icpk
    tcp3 = lcpk + icpk * min(1.0, dim(tice, tz) / (tice - t_wfr))

    if p1 >= p_min:

        # Instant deposit all water vapor to cloud ice when temperature is super low
        if tz < t_min:

            sink = dim(1.0e-7, qvz)
            qvz = qvz - sink
            qiz = qiz + sink
            q_sol = q_sol + sink
            cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            tz = tz + sink * (lhl + lhi) / cvm

            if do_qa == 0:
                qaz = qaz + 1.0  # Air fully saturated; 100% cloud cover

        else:

            # Update heat capacity and latent heat coefficient
            lhl = lv00 + d0_vap * tz
            lhi = li00 + dc_ice * tz
            lcpk = lhl / cvm
            icpk = lhi / cvm
            tcpk = lcpk + icpk
            tcp3 = lcpk + icpk * min(1.0, dim(tice, tz) / (tice - t_wfr))

            # Instant evaporation / sublimation of all clouds if rh < rh_adj -- > cloud free
            qpz = qvz + qlz + qiz
            tin = tz - (lhl * (qlz + qiz) + lhi * qiz) / (
                c_air + qpz * c_vap + qrz * c_liq + (qsz + qgz) * c_ice
            )

            if tin > t_sub + 6.0:

                rh = qpz / iqs1(tin, den)

                if rh < rh_adj:  # qpz / rh_adj < qs

                    tz = tin
                    qvz = qpz
                    qlz = 0.0
                    qiz = 0.0

            if ((tin > t_sub + 6.0) and (rh >= rh_adj)) or (tin <= t_sub + 6.0):

                # Cloud water < -- > vapor adjustment
                qsw, dwsdt = wqs2(tz, den)

                dq0 = qsw - qvz

                if dq0 > 0.0:

                    # Added ql factor to prevent the situation of high ql and low RH
                    factor = min(
                        1.0, fac_l2v * (10.0 * dq0 / qsw)
                    )  # The rh dependent factor = 1 at 90%
                    evap = min(qlz, factor * dq0 / (1.0 + tcp3 * dwsdt))

                else:  # Condensate all excess vapor into cloud water

                    evap = dq0 / (1.0 + tcp3 * dwsdt)

                qvz = qvz + evap
                qlz = qlz - evap
                q_liq = q_liq - evap
                cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz = tz - evap * lhl / cvm

                # Update heat capacity and latent heat coefficient
                lhi = li00 + dc_ice * tz
                icpk = lhi / cvm

                # Enforce complete freezing below -48 degrees Celsius
                dtmp = t_wfr - tz  # [-40, -48]

                if (dtmp > 0.0) and (qlz > qcmin):

                    sink = min(qlz, min(qlz * dtmp * 0.125, dtmp / icpk))
                    qlz = qlz - sink
                    qiz = qiz + sink
                    q_liq = q_liq - sink
                    q_sol = q_sol + sink
                    cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz = tz + sink * lhi / cvm

                # Update heat capacity and latent heat coefficient
                lhi = li00 + dc_ice * tz
                icpk = lhi / cvm

                # Bigg mechanism
                if fast_sat_adj == 1:

                    dt_pisub = 0.5 * dts

                else:

                    dt_pisub = dts

                    tc = tice - tz

                    if (qlz > qrmin) and (tc > 0.0):

                        sink = (
                            3.3333e-10 * dts * (exp(0.66 * tc) - 1.0) * den * qlz * qlz
                        )
                        sink = min(qlz, min(tc / icpk, sink))
                        qlz = qlz - sink
                        qiz = qiz + sink
                        q_liq = q_liq - sink
                        q_sol = q_sol + sink
                        cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                        tz = tz + sink * lhi / cvm

                # Update capacity heat and latent heat coefficient
                lhl = lv00 + d0_vap * tz
                lhi = li00 + dc_ice * tz
                lcpk = lhl / cvm
                icpk = lhi / cvm
                tcpk = lcpk + icpk

                # Sublimation / deposition of ice
                if tz < tice:

                    qsi, dqsdt = iqs2(tz, den)

                    dq = qvz - qsi
                    sink = dq / (1.0 + tcpk * dqsdt)

                    if qiz > qrmin:

                        # - Eq 9, hong et al. 2004, mwr
                        # - For a and b, see dudhia 1989: page 3103 eq (b7) and (b8)
                        pidep = (
                            dt_pisub
                            * dq
                            * 349138.78
                            * exp(0.875 * log(qiz * den))
                            / (
                                qsi * den * lat2 / (0.0243 * rvgas * tz ** 2)
                                + 4.42478e4
                            )
                        )

                    else:

                        pidep = 0.0

                    if dq > 0.0:  # Vapor -- > ice

                        tmp = tice - tz

                        # The following should produce more ice at higher altitude
                        qi_crt = qi_gen * min(qi_lim, 0.1 * tmp) / den
                        sink = min(sink, min(max(qi_crt - qiz, pidep), tmp / tcpk))

                    else:  # Ice -- > vapor

                        pidep = pidep * min(1.0, dim(tz, t_sub) * 0.2)
                        sink = max(pidep, max(sink, -qiz))

                    qvz = qvz - sink
                    qiz = qiz + sink
                    q_sol = q_sol + sink
                    cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz = tz + sink * (lhl + lhi) / cvm

                # Update capacity heat and latent heat coefficient
                lhl = lv00 + d0_vap * tz
                lhi = li00 + dc_ice * tz
                lcpk = lhl / cvm
                icpk = lhi / cvm
                tcpk = lcpk + icpk

                # - Sublimation / deposition of snow
                # - This process happens for the whole temperature range
                if qsz > qrmin:

                    qsi, dqsdt = iqs2(tz, den)

                    qden = qsz * den
                    tmp = exp(0.65625 * log(qden))
                    tsq = tz * tz
                    dq = (qsi - qvz) / (1.0 + tcpk * dqsdt)
                    pssub = (
                        cssub_0
                        * tsq
                        * (cssub_1 * sqrt(qden) + cssub_2 * tmp * sqrt(denfac))
                        / (cssub_3 * tsq + cssub_4 * qsi * den)
                    )
                    pssub = (qsi - qvz) * dts * pssub

                    if pssub > 0.0:  # qs -- > qv, sublimation

                        pssub = min(pssub * min(1.0, dim(tz, t_sub) * 0.2), qsz)

                    else:

                        if tz > tice:

                            pssub = 0.0  # No deposition

                        else:

                            pssub = max(pssub, max(dq, (tz - tice) / tcpk))

                    qsz = qsz - pssub
                    qvz = qvz + pssub
                    q_sol = q_sol - pssub
                    cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz = tz - pssub * (lhl + lhi) / cvm

                # Update capacity heat and latent heat coefficient
                lhl = lv00 + d0_vap * tz
                lhi = li00 + dc_ice * tz
                lcpk = lhl / cvm
                icpk = lhi / cvm
                tcpk = lcpk + icpk

                # Simplified 2-way grapuel sublimation-deposition mechanism
                if qgz > qrmin:

                    qsi, dqsdt = iqs2(tz, den)

                    dq = (qvz - qsi) / (1.0 + tcpk * dqsdt)
                    pgsub = (qvz / qsi - 1.0) * qgz

                    if pgsub > 0.0:  # Deposition

                        if tz > tice:

                            pgsub = 0.0

                        else:

                            pgsub = min(
                                min(fac_v2g * pgsub, 0.2 * dq),
                                min(qlz + qrz, (tice - tz) / tcpk),
                            )

                    else:  # Sublimation

                        pgsub = max(fac_g2v * pgsub, dq) * min(
                            1.0, dim(tz, t_sub) * 0.1
                        )

                    qgz = qgz + pgsub
                    qvz = qvz - pgsub
                    q_sol = q_sol + pgsub
                    cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz = tz + pgsub * (lhl + lhi) / cvm

                """
                USE_MIN_EVAP
                """
                # Update capacity heat and latent heat coefficient
                lhl = lv00 + d0_vap * tz
                lcpk = lhl / cvm

                # Minimum evap of rain in dry environmental air
                if qrz > qcmin:

                    qsw, dqsdt = wqs2(tz, den)

                    sink = min(qrz, dim(rh_rain * qsw, qvz) / (1.0 + lcpk * dqsdt))
                    qvz = qvz + sink
                    qrz = qrz - sink
                    q_liq = q_liq - sink
                    cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                    tz = tz - sink * lhl / cvm
                """
                END USE_MIN_EVAP
                """

                # Update capacity heat and latent heat coefficient
                lhl = lv00 + d0_vap * tz
                cvm = c_air + (qvz + q_liq + q_sol) * c_vap
                lcpk = lhl / cvm

                # Compute cloud fraction
                # Combine water species
                if do_qa == 0:

                    if rad_snow == 1:
                        q_sol = qiz + qsz
                    else:
                        q_sol = qiz

                    if rad_rain == 1:
                        q_liq = qlz + qrz
                    else:
                        q_liq = qlz

                    q_cond = q_liq + q_sol

                    qpz = qvz + q_cond  # qpz is conserved

                    # Use the "liquid-frozen water temperature" (tin) to compute saturated specific humidity
                    tin = tz - (lcpk * q_cond + icpk * q_sol)  # Minimum temperature

                    # Determine saturated specific humidity
                    """
                    THIS NEEDS TO BE DONE IN ORDER FOR THE CODE TO RUN !?!?
                    """
                    t_wfr_tmp = t_wfr
                    if tin <= t_wfr:

                        # Ice phase
                        qstar = iqs1(tin, den)

                    elif tin >= tice:

                        # Liquid phase
                        qstar = wqs1(tin, den)

                    else:

                        # Mixed phase
                        qsi = iqs1(tin, den)
                        qsw = wqs1(tin, den)

                        if q_cond > 3.0e-6:

                            rqi = q_sol / q_cond

                        else:

                            # Mostly liquid water q_cond (k) at initial cloud development stage
                            """
                            THIS NEEDS TO BE DONE IN ORDER FOR THE CODE TO RUN !?!?
                            rqi = (tice - tin) / (tice - t_wfr)
                            """
                            rqi = (tice - tin) / (tice - t_wfr_tmp)

                        qstar = rqi * qsi + (1.0 - rqi) * qsw

                    # Assuming subgrid linear distribution in horizontal; this is
                    # effectively a smoother for the binary cloud scheme
                    if qpz > qrmin:

                        # Partial cloudiness by pdf
                        dq = max(qcmin, h_var * qpz)
                        q_plus = qpz + dq  # Cloud free if qstar > q_plus
                        q_minus = qpz - dq

                        if qstar < q_minus:

                            qaz = qaz + 1.0  # Air fully saturated; 100% cloud cover

                        elif (qstar < q_plus) and (q_cond > qc_crt):

                            qaz = qaz + (q_plus - qstar) / (
                                dq + dq
                            )  # Partial cloud cover

    return qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz


@gtscript.function
def icloud_main(
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
):

    # Update capacity heat and latent heat coefficient
    lhi = li00 + dc_ice * tz
    icpk = lhi / cvm

    if p1 >= p_min:

        pgacr = 0.0
        pgacw = 0.0

        tc = tz - tice

        if tc >= 0.0:

            # Melting of snow
            dqs0 = ces0 / p1 - qvz

            if qsz > qcmin:

                # psacw: accretion of cloud water by snow (only rate is used (for
                # snow melt) since tc > 0.)
                if qlz > qrmin:

                    factor = denfac * csacw * exp(0.8125 * log(qsz * den))
                    psacw = factor / (1.0 + dts * factor) * qlz  # Rate

                else:

                    psacw = 0.0

                # psacr: accretion of rain by melted snow
                # pracs: accretion of snow by rain
                if qrz > qrmin:

                    psacr = min(
                        acr3d(
                            vtsz, vtrz, qrz, qsz, csacr, acco_01, acco_11, acco_21, den
                        ),
                        qrz * rdts,
                    )
                    pracs = acr3d(
                        vtrz, vtsz, qsz, qrz, cracs, acco_00, acco_10, acco_20, den
                    )

                else:

                    psacr = 0.0
                    pracs = 0.0

                # Total snow sink
                # psmlt: snow melt (due to rain accretion)
                psmlt = max(
                    0.0,
                    smlt(
                        tc,
                        dqs0,
                        qsz * den,
                        psacw,
                        psacr,
                        csmlt_0,
                        csmlt_1,
                        csmlt_2,
                        csmlt_3,
                        csmlt_4,
                        den,
                        denfac,
                    ),
                )
                sink = min(qsz, min(dts * (psmlt + pracs), tc / icpk))
                qsz = qsz - sink
                tmp = min(sink, dim(qs_mlt, qlz))  # Maximum ql due to snow melt
                qlz = qlz + tmp
                qrz = qrz + sink - tmp
                q_liq = q_liq + sink
                q_sol = q_sol - sink
                cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz = tz - sink * lhi / cvm
                tc = tz - tice

            # Update capacity heat and latent heat coefficient
            lhi = li00 + dc_ice * tz
            icpk = lhi / cvm

            # Melting of graupel
            if (qgz > qcmin) and (tc > 0.0):

                # pgacr: accretion of rain by graupel
                if qrz > qrmin:

                    pgacr = min(
                        acr3d(
                            vtgz, vtrz, qrz, qgz, cgacr, acco_02, acco_12, acco_22, den
                        ),
                        rdts * qrz,
                    )

                # pgacw: accretion of cloud water by graupel
                qden = qgz * den

                if qlz > qrmin:

                    factor = cgacw * qden / sqrt(den * sqrt(sqrt(qden)))
                    pgacw = factor / (1.0 + dts * factor) * qlz  # Rate

                # pgmlt: graupel melt
                pgmlt = dts * gmlt(
                    tc,
                    dqs0,
                    qden,
                    pgacw,
                    pgacr,
                    cgmlt_0,
                    cgmlt_1,
                    cgmlt_2,
                    cgmlt_3,
                    cgmlt_4,
                    den,
                )
                pgmlt = min(max(0.0, pgmlt), min(qgz, tc / icpk))
                qgz = qgz - pgmlt
                qrz = qrz + pgmlt
                q_liq = q_liq + pgmlt
                q_sol = q_sol - pgmlt
                cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz = tz - pgmlt * lhi / cvm

        else:

            # Cloud ice proc
            # psaci: accretion of cloud ice by snow
            if qiz > 3.0e-7:  # Cloud ice sink terms

                if qsz > 1.0e-7:

                    # sjl added (following lin eq. 23) the temperature dependency to
                    # reduce accretion, use esi = exp(0.05 * tc) as in hong et al 2004
                    factor = (
                        dts * denfac * csaci * exp(0.05 * tc + 0.8125 * log(qsz * den))
                    )
                    psaci = factor / (1.0 + factor) * qiz

                else:

                    psaci = 0.0

                # pasut: autoconversion: cloud ice -- > snow
                # - Similar to lfo 1983: eq. 21 solved implicitly
                # - Threshold from wsm6 scheme, hong et al 2004, eq (13) : qi0_crt ~0.8e-4
                qim = qi0_crt / den

                # - Assuming linear subgrid vertical distribution of cloud ice
                # - The mismatch computation following lin et al. 1994, mwr
                if const_vi == 1:

                    tmp = fac_i2s

                else:

                    tmp = fac_i2s * exp(0.025 * tc)

                di = max(di, qrmin)
                q_plus = qiz + di

                if q_plus > (qim + qrmin):

                    if qim > (qiz - di):

                        dq = (0.25 * (q_plus - qim) ** 2) / di

                    else:

                        dq = qiz - qim

                    psaut = tmp * dq

                else:

                    psaut = 0.0

                # sink is no greater than 75% of qi
                sink = min(0.75 * qiz, psaci + psaut)
                qiz = qiz - sink
                qsz = qsz + sink

                # pgaci: accretion of cloud ice by graupel
                if qgz > 1.0e-6:

                    # - factor = dts * cgaci / sqrt (den (k)) * exp (0.05 * tc + 0.875 * log (qg * den (k)))
                    # - Simplified form: remove temp dependency & set the exponent "0.875" -- > 1
                    factor = dts * cgaci * sqrt(den) * qgz
                    pgaci = factor / (1.0 + factor) * qiz
                    qiz = qiz - pgaci
                    qgz = qgz + pgaci

            # Cold-rain proc
            # Rain to ice, snow, graupel processes
            tc = tz - tice

            if (qrz > 1e-7) and (tc < 0.0):

                # - Sink terms to qr: psacr + pgfr
                # - Source terms to qs: psacr
                # - Source terms to qg: pgfr
                # psacr accretion of rain by snow
                if qsz > 1.0e-7:  # If snow exists

                    psacr = dts * acr3d(
                        vtsz, vtrz, qrz, qsz, csacr, acco_01, acco_11, acco_21, den
                    )

                else:

                    psacr = 0.0

                # pgfr: rain freezing -- > graupel
                pgfr = (
                    dts
                    * cgfr_0
                    / den
                    * (exp(-cgfr_1 * tc) - 1.0)
                    * exp(1.75 * log(qrz * den))
                )

                # Total sink to qr
                sink = psacr + pgfr
                factor = min(sink, min(qrz, -tc / icpk)) / max(sink, qrmin)

                psacr = factor * psacr
                pgfr = factor * pgfr

                sink = psacr + pgfr
                qrz = qrz - sink
                qsz = qsz + psacr
                qgz = qgz + pgfr
                q_liq = q_liq - sink
                q_sol = q_sol + sink
                cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz = tz + sink * lhi / cvm

            # Update capacity heat and latent heat coefficient
            lhi = li00 + dc_ice * tz
            icpk = lhi / cvm

            # Graupel production terms
            if qsz > 1.0e-7:

                # Accretion: snow -- > graupel
                if qgz > qrmin:

                    sink = dts * acr3d(
                        vtgz, vtsz, qsz, qgz, cgacs, acco_03, acco_13, acco_23, den
                    )

                else:

                    sink = 0.0

                # Autoconversion snow -- > graupel
                qsm = qs0_crt / den

                if qsz > qsm:

                    factor = dts * 1.0e-3 * exp(0.09 * (tz - tice))
                    sink = sink + factor / (1.0 + factor) * (qsz - qsm)

                sink = min(qsz, sink)
                qsz = qsz - sink
                qgz = qgz + sink

            if (qgz > 1.0e-7) and (tz < tice0):

                # pgacw: accretion of cloud water by graupel
                if qlz > 1.0e-6:

                    qden = qgz * den
                    factor = dts * cgacw * qden / sqrt(den * sqrt(sqrt(qden)))
                    pgacw = factor / (1.0 + factor) * qlz

                else:

                    pgacw = 0.0

                # pgacr: accretion of rain by graupel
                if qrz > 1.0e-6:

                    pgacr = min(
                        dts
                        * acr3d(
                            vtgz, vtrz, qrz, qgz, cgacr, acco_02, acco_12, acco_22, den
                        ),
                        qrz,
                    )

                else:

                    pgacr = 0.0

                sink = pgacr + pgacw
                factor = min(sink, dim(tice, tz) / icpk) / max(sink, qrmin)
                pgacr = factor * pgacr
                pgacw = factor * pgacw

                sink = pgacr + pgacw
                qgz = qgz + sink
                qrz = qrz - pgacr
                qlz = qlz - pgacw
                q_liq = q_liq - sink
                q_sol = q_sol + sink
                cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz = tz + sink * lhi / cvm

    qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz = subgrid_z_proc(
        c_air,
        c_vap,
        d0_vap,
        lv00,
        cssub_0,
        cssub_1,
        cssub_2,
        cssub_3,
        cssub_4,
        t_wfr,
        dts,
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
        p1,
    )

    return qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz
