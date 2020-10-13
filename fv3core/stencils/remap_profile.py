import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.profile_limiters as limiters
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


def grid():
    return spec.grid


@gtscript.function
def limit_minmax(q, a4):
    tmp = a4[0, 0, -1] if a4[0, 0, -1] > a4 else a4
    ret = q if q < tmp else tmp
    return ret


@gtscript.function
def limit_maxmin(q, a4):
    tmp2 = a4[0, 0, -1] if a4[0, 0, -1] < a4 else a4
    ret = q if q > tmp2 else tmp2
    return ret


@gtscript.function
def limit_both(q, a4):
    ret = limit_minmax(q, a4)
    ret = limit_maxmin(ret, a4)
    return ret


@gtscript.function
def constrain_interior(q, gam, a4):
    return (
        limit_both(q, a4)
        if (gam[0, 0, -1] * gam[0, 0, 1] > 0.0)
        else limit_maxmin(q, a4)
        if (gam[0, 0, -1] > 0.0)
        else limit_minmax(q, a4)
    )


@gtstencil()
def set_vals_2(gam: sd, q: sd, delp: sd, a4_1: sd, q_bot: sd, qs: sd):
    with computation(PARALLEL):
        with interval(0, 1):
            # set top
            # gam = 0.5
            q = 1.5 * a4_1
    with computation(FORWARD):
        with interval(1, 2):
            gam = 0.5
            grid_ratio = delp[0, 0, -1] / delp
            bet = 2.0 + grid_ratio + grid_ratio - gam
            q = (3.0 * (a4_1[0, 0, -1] + a4_1) - q[0, 0, -1]) / bet
        with interval(2, -2):
            # set middle
            old_grid_ratio = delp[0, 0, -2] / delp[0, 0, -1]
            old_bet = 2.0 + old_grid_ratio + old_grid_ratio - gam[0, 0, -1]
            gam = old_grid_ratio / old_bet
            grid_ratio = delp[0, 0, -1] / delp
            bet = 2.0 + grid_ratio + grid_ratio - gam
            q = (3.0 * (a4_1[0, 0, -1] + a4_1) - q[0, 0, -1]) / bet
            # gam[0, 0, 1] = grid_ratio / bet
    with computation(FORWARD):
        with interval(-2, -1):
            # set bottom
            old_grid_ratio = delp[0, 0, -2] / delp[0, 0, -1]
            old_bet = 2.0 + old_grid_ratio + old_grid_ratio - gam[0, 0, -1]
            gam = old_grid_ratio / old_bet
            grid_ratio = delp[0, 0, -1] / delp
            q = (
                3.0 * (a4_1[0, 0, -1] + a4_1) - grid_ratio * qs[0, 0, 1] - q[0, 0, -1]
            ) / (2.0 + grid_ratio + grid_ratio - gam)
            q_bot = qs
    with computation(PARALLEL):
        with interval(-1, None):
            q_bot = qs
            q = qs
    with computation(BACKWARD), interval(0, -2):
        q = q - gam[0, 0, 1] * q[0, 0, 1]


@gtstencil()
def set_vals_1(gam: sd, q: sd, delp: sd, a4_1: sd, q_bot: sd):
    with computation(PARALLEL):
        with interval(0, 1):
            # set top
            grid_ratio = delp[0, 0, 1] / delp
            bet = grid_ratio * (grid_ratio + 0.5)
            q = (
                (grid_ratio + grid_ratio) * (grid_ratio + 1.0) * a4_1 + a4_1[0, 0, 1]
            ) / bet
            gam = (1.0 + grid_ratio * (grid_ratio + 1.5)) / bet
    with computation(FORWARD):
        with interval(1, -1):
            # set middle
            d4 = delp[0, 0, -1] / delp
            bet = 2.0 + d4 + d4 - gam[0, 0, -1]
            q = (3.0 * (a4_1[0, 0, -1] + d4 * a4_1) - q[0, 0, -1]) / bet
            gam = d4 / bet
    with computation(PARALLEL):
        with interval(-1, None):
            # set bottom
            d4 = delp[0, 0, -2] / delp[0, 0, -1]
            a_bot = 1.0 + d4 * (d4 + 1.5)
            q = (
                2.0 * d4 * (d4 + 1.0) * a4_1[0, 0, -1]
                + a4_1[0, 0, -2]
                - a_bot * q[0, 0, -1]
            ) / (d4 * (d4 + 0.5) - a_bot * gam[0, 0, -1])
    with computation(BACKWARD), interval(0, -1):
        q = q - gam * q[0, 0, 1]


@gtstencil()
def set_avals(q: sd, a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd, q_bot: sd):
    with computation(PARALLEL):
        with interval(0, -1):
            a4_2 = q
            a4_3 = q[0, 0, 1]
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
    with computation(PARALLEL):
        with interval(-1, None):
            a4_2 = q
            a4_3 = q_bot
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def Apply_constraints(q: sd, gam: sd, a4_1: sd, a4_2: sd, a4_3: sd, iv: int):
    with computation(PARALLEL):
        with interval(1, None):
            a4_1_0 = a4_1[0, 0, -1]
            tmp = a4_1_0 if a4_1_0 > a4_1 else a4_1
            tmp2 = a4_1_0 if a4_1_0 < a4_1 else a4_1
            gam = a4_1 - a4_1_0
        with interval(1, 2):
            # do top
            q = q if q < tmp else tmp
            q = q if q > tmp2 else tmp2
    with computation(FORWARD):
        with interval(2, -1):
            # do middle
            if (gam[0, 0, -1] * gam[0, 0, 1]) > 0:
                q = q if q < tmp else tmp
                q = q if q > tmp2 else tmp2
            elif gam[0, 0, -1] > 0:
                # there's a local maximum
                q = q if q > tmp2 else tmp2
            else:
                # there's a local minimum
                q = q if q < tmp else tmp
                q = 0.0 if (q < 0.0 and iv == 0) else q
            # q = constrain_interior(q, gam, a4_1)
        with interval(-1, None):
            # do bottom
            q = q if q < tmp else tmp
            q = q if q > tmp2 else tmp2
    with computation(PARALLEL):
        with interval(...):
            # re-set a4_2 and a4_3
            a4_2 = q
            a4_3 = q[0, 0, 1]


@gtstencil()
def set_extm(extm: sd, a4_1: sd, a4_2: sd, a4_3: sd, gam: sd):
    with computation(PARALLEL):
        with interval(0, 1):
            extm = (a4_2 - a4_1) * (a4_3 - a4_1) > 0.0
        with interval(1, -1):
            extm = gam * gam[0, 0, 1] < 0.0
        with interval(-1, None):
            extm = (a4_2 - a4_1) * (a4_3 - a4_1) > 0.0


@gtstencil()
def set_exts(a4_4: sd, ext5: sd, ext6: sd, a4_1: sd, a4_2: sd, a4_3: sd):
    with computation(PARALLEL), interval(...):
        x0 = 2.0 * a4_1 - (a4_2 + a4_3)
        x1 = abs(a4_2 - a4_3)
        a4_4 = 3.0 * x0
        ext5 = abs(x0) > x1
        ext6 = abs(a4_4) > x1


@gtstencil()
def set_top_as_iv0(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL):
        with interval(0, 1):
            a4_2 = a4_2 if a4_2 > 0.0 else 0.0
    with computation(PARALLEL):
        with interval(...):
            a4_4 = 3 * (2 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def set_top_as_iv1(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL):
        with interval(0, 1):
            a4_2 = 0.0 if a4_2 * a4_1 <= 0.0 else a4_2
    with computation(PARALLEL):
        with interval(...):
            a4_4 = 3 * (2 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def set_top_as_iv2(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL):
        with interval(0, 1):
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
    with computation(PARALLEL):
        with interval(1, None):
            a4_4 = 3 * (2 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def set_top_as_else(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL):
        with interval(...):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def set_inner_as_kordsmall(
    a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd, gam: sd, extm: sd, ext5: sd, ext6: sd
):
    with computation(PARALLEL), interval(...):
        # left edges?
        pmp_1 = a4_1 - gam[0, 0, 1]
        lac_1 = pmp_1 + 1.5 * gam[0, 0, 2]
        tmp_min = (
            a4_1
            if (a4_1 < pmp_1) and (a4_1 < lac_1)
            else pmp_1
            if pmp_1 < lac_1
            else lac_1
        )
        tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
        tmp_max = (
            a4_1
            if (a4_1 > pmp_1) and (a4_1 > lac_1)
            else pmp_1
            if pmp_1 > lac_1
            else lac_1
        )
        a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
        # right edges?
        pmp_2 = a4_1 + 2.0 * gam[0, 0, 1]
        lac_2 = pmp_2 - 1.5 * gam[0, 0, -1]
        tmp_min = (
            a4_1
            if (a4_1 < pmp_2) and (a4_1 < lac_2)
            else pmp_2
            if pmp_2 < lac_2
            else lac_2
        )
        tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
        tmp_max = (
            a4_1
            if (a4_1 > pmp_2) and (a4_1 > lac_2)
            else pmp_2
            if pmp_2 > lac_2
            else lac_2
        )
        a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
        a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def set_inner_as_kord9(
    a4_1: sd,
    a4_2: sd,
    a4_3: sd,
    a4_4: sd,
    gam: sd,
    extm: sd,
    ext5: sd,
    ext6: sd,
    qmin: float,
):
    with computation(PARALLEL), interval(...):
        pmp_1 = a4_1 - 2.0 * gam[0, 0, 1]
        lac_1 = pmp_1 + 1.5 * gam[0, 0, 2]
        pmp_2 = a4_1 + 2.0 * gam
        lac_2 = pmp_2 - 1.5 * gam[0, 0, -1]
        tmp_min = a4_1
        tmp_max = a4_2
        tmp_max0 = a4_1
        diff_23 = 0.0
        if extm and extm[0, 0, -1]:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        elif extm and extm[0, 0, 1]:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        elif extm > 0.0 and (qmin > 0.0 and a4_1 < qmin):
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            diff_23 = a4_2 - a4_3
            a4_4 = 6.0 * a4_1 - 3.0 * (a4_2 + a4_3)
            if abs(a4_4) > abs(diff_23):
                tmp_min = (
                    a4_1
                    if (a4_1 < pmp_1) and (a4_1 < lac_1)
                    else pmp_1
                    if pmp_1 < lac_1
                    else lac_1
                )
                tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
                tmp_max = (
                    a4_1
                    if (a4_1 > pmp_1) and (a4_1 > lac_1)
                    else pmp_1
                    if pmp_1 > lac_1
                    else lac_1
                )
                a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                tmp_min = (
                    a4_1
                    if (a4_1 < pmp_2) and (a4_1 < lac_2)
                    else pmp_2
                    if pmp_2 < lac_2
                    else lac_2
                )
                tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
                tmp_max = (
                    a4_1
                    if (a4_1 > pmp_2) and (a4_1 > lac_2)
                    else pmp_2
                    if pmp_2 > lac_2
                    else lac_2
                )
                a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                a4_4 = 6.0 * a4_1 - 3.0 * (a4_2 + a4_3)
            else:
                a4_2 = a4_2


@gtstencil()
def set_inner_as_kord10(
    a4_1: sd,
    a4_2: sd,
    a4_3: sd,
    a4_4: sd,
    gam: sd,
    extm: sd,
    ext5: sd,
    ext6: sd,
    pmp_2: sd,
    lac_2: sd,
    tmp_min3: sd,
    tmp_max3: sd,
    tmp3: sd,
):
    with computation(PARALLEL), interval(...):
        pmp_1 = a4_1 - 2.0 * gam[0, 0, 1]
        lac_1 = pmp_1 + 1.5 * gam[0, 0, 2]
        pmp_2 = a4_1 + 2.0 * gam
        lac_2 = pmp_2 - 1.5 * gam[0, 0, -1]
        tmp_min2 = (
            a4_1
            if (a4_1 < pmp_1) and (a4_1 < lac_1)
            else pmp_1
            if pmp_1 < lac_1
            else lac_1
        )
        tmp_max2 = (
            a4_1
            if (a4_1 > pmp_1) and (a4_1 > lac_1)
            else pmp_1
            if pmp_1 > lac_1
            else lac_1
        )
        tmp2 = a4_2 if a4_2 > tmp_min2 else tmp_min2

        # tmp_min3 = (
        #    a4_1
        #    if (a4_1 < pmp_2) and (a4_1 < lac_2)
        #    else pmp_2
        #    if pmp_2 < lac_2
        #    else lac_2
        # )
        # tmp_max3 = (
        #    a4_1
        #    if (a4_1 > pmp_2) and (a4_1 > lac_2)
        #    else pmp_2
        #    if pmp_2 > lac_2
        #    else lac_2
        # )
        tmp_min3 = a4_1 if a4_1 < pmp_2 else pmp_2
        tmp_min3 = lac_2 if lac_2 < tmp_min3 else tmp_min3
        tmp_max3 = a4_1 if a4_1 > pmp_2 else pmp_2
        tmp_max3 = lac_2 if lac_2 > tmp_max3 else tmp_max3
        tmp3 = a4_3 if a4_3 > tmp_min3 else tmp_min3
        if ext5:
            if ext5[0, 0, -1] or ext5[0, 0, 1]:
                a4_2 = a4_1
                a4_3 = a4_1
            elif ext6[0, 0, -1] or ext6[0, 0, 1]:
                a4_2 = tmp2 if tmp2 < tmp_max2 else tmp_max2
                a4_3 = tmp3 if tmp3 < tmp_max3 else tmp_max3
            else:
                a4_2 = a4_2
        elif ext6:
            if ext5[0, 0, -1] or ext5[0, 0, 1]:
                a4_2 = tmp2 if tmp2 < tmp_max2 else tmp_max2
                a4_3 = tmp3 if tmp3 < tmp_max3 else tmp_max3
            else:
                a4_2 = a4_2
        else:
            a4_2 = a4_2
    with computation(PARALLEL), interval(...):
        a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def set_bottom_as_iv0(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL):
        with interval(1, None):
            a4_3 = a4_3 if a4_3 > 0.0 else 0.0
    with computation(PARALLEL):
        with interval(...):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def set_bottom_as_iv1(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL):
        with interval(-1, None):
            a4_3 = 0.0 if a4_3 * a4_1 <= 0.0 else a4_3
    with computation(PARALLEL):
        with interval(...):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


@gtstencil()
def set_bottom_as_else(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL):
        with interval(...):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


def compute(qs, a4_1, a4_2, a4_3, a4_4, delp, km, i1, i2, iv, kord, jslice, qmin=0.0):
    i_extent = i2 - i1 + 1
    j_extent = jslice.stop - jslice.start
    js = jslice.start
    grid = spec.grid
    orig = (i1, js, 0)
    full_orig = (grid.is_, js, 0)
    dom = (i_extent, j_extent, km)
    gam = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    q = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    q_bot = utils.make_storage_from_shape(delp.shape, origin=full_orig)

    qs_field = utils.make_storage_from_shape(delp.shape, origin=full_orig)

    qs_field[i1 : i2 + 1, js : js + j_extent, -1] = qs[
        i1 : i2 + 1, js : js + j_extent, 0
    ]  # make a qs that can be passed to a stencil

    extm = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    ext5 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    ext6 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    pmp_2 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    lac_2 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    tmp_min3 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    tmp_max3 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    tmp3 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    if iv == -2:
        set_vals_2(
            gam,
            q,
            delp,
            a4_1,
            q_bot,
            qs_field,
            origin=orig,
            domain=(i_extent, j_extent, km + 1),
        )
    else:
        set_vals_1(
            gam, q, delp, a4_1, q_bot, origin=orig, domain=(i_extent, j_extent, km + 1)
        )

    if abs(kord) > 16:
        set_avals(q, a4_1, a4_2, a4_3, a4_4, q_bot, origin=orig, domain=dom)
    else:
        Apply_constraints(q, gam, a4_1, a4_2, a4_3, iv, origin=orig, domain=dom)
        set_extm(extm, a4_1, a4_2, a4_3, gam, origin=orig, domain=dom)
        if abs(kord) > 9:
            set_exts(a4_4, ext5, ext6, a4_1, a4_2, a4_3, origin=orig, domain=dom)
        if iv == 0:
            set_top_as_iv0(
                a4_1, a4_2, a4_3, a4_4, origin=orig, domain=(i_extent, j_extent, 2)
            )
            a4_1, a4_2, a4_3, a4_4 = limiters.compute(
                a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 0, 1, js, j_extent
            )
        elif iv == -1:
            set_top_as_iv1(
                a4_1, a4_2, a4_3, a4_4, origin=orig, domain=(i_extent, j_extent, 2)
            )

            a4_1, a4_2, a4_3, a4_4 = limiters.compute(
                a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 0, 1, js, j_extent
            )

        elif iv == 2:
            set_top_as_iv2(
                a4_1, a4_2, a4_3, a4_4, origin=orig, domain=(i_extent, j_extent, 2)
            )
        else:
            set_top_as_else(
                a4_1, a4_2, a4_3, a4_4, origin=orig, domain=(i_extent, j_extent, 2)
            )
            a4_1, a4_2, a4_3, a4_4 = limiters.compute(
                a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 0, 1, js, j_extent
            )
        a4_1, a4_2, a4_3, a4_4 = limiters.compute(
            a4_1, a4_2, a4_3, a4_4, extm, 2, i1, i_extent, 1, 1, js, j_extent
        )
        if abs(kord) < 9:
            print("WARNING: Only kord=10 and 9 have been tested.")
            set_inner_as_kordsmall(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                gam,
                extm,
                ext5,
                ext6,
                origin=(i1, js, 2),
                domain=(i_extent, j_extent, km - 4),
            )
        elif abs(kord) == 9:
            set_inner_as_kord9(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                gam,
                extm,
                ext5,
                ext6,
                qmin,
                origin=(i1, js, 2),
                domain=(i_extent, j_extent, km - 4),
            )
        elif abs(kord) == 10:
            set_inner_as_kord10(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                gam,
                extm,
                ext5,
                ext6,
                pmp_2,
                lac_2,
                tmp_min3,
                tmp_max3,
                tmp3,
                origin=(i1, js, 2),
                domain=(i_extent, j_extent, km - 4),
            )
        else:
            raise Exception("kord {0} not implemented.".format(kord))

        if iv == 0:
            a4_1, a4_2, a4_3, a4_4 = limiters.compute(
                a4_1, a4_2, a4_3, a4_4, extm, 0, i1, i_extent, 2, km - 4, js, j_extent
            )

        if iv == 0:
            set_bottom_as_iv0(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                origin=(i1, js, km - 2),
                domain=(i_extent, j_extent, 2),
            )
        elif iv == -1:
            set_bottom_as_iv1(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                origin=(i1, js, km - 2),
                domain=(i_extent, j_extent, 2),
            )
        else:
            set_bottom_as_else(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                origin=(i1, js, km - 2),
                domain=(i_extent, j_extent, 2),
            )
        a4_1, a4_2, a4_3, a4_4 = limiters.compute(
            a4_1, a4_2, a4_3, a4_4, extm, 2, i1, i_extent, km - 2, 1, js, j_extent
        )
        a4_1, a4_2, a4_3, a4_4 = limiters.compute(
            a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, km - 1, 1, js, j_extent
        )

    return a4_1, a4_2, a4_3, a4_4
