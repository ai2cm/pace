from typing import Tuple

import gt4py.gtscript as gtscript
from gt4py.gtscript import __INLINED, BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.utils.typing import FloatField, FloatFieldIJ


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


@gtscript.function
def remap_constraint(
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    extm: FloatField,
    iv: int,
):
    # posdef_constraint_iv0
    if iv == 0:
        if a4_1 <= 0.0:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            if abs(a4_3 - a4_2) < -a4_4:
                if (
                    a4_1 + 0.25 * (a4_3 - a4_2) ** 2 / a4_4 + a4_4 * (1.0 / 12.0)
                ) < 0.0:
                    if (a4_1 < a4_3) and (a4_1 < a4_2):
                        a4_3 = a4_1
                        a4_2 = a4_1
                        a4_4 = 0.0
                    elif a4_3 > a4_2:
                        a4_4 = 3.0 * (a4_2 - a4_1)
                        a4_3 = a4_2 - a4_4
                    else:
                        a4_4 = 3.0 * (a4_3 - a4_1)
                        a4_2 = a4_3 - a4_4
    if iv == 1:
        # posdef_constraint_iv1
        da1 = a4_3 - a4_2
        da2 = da1 * da1
        a6da = a4_4 * da1
        if ((a4_1 - a4_2) * (a4_1 - a4_3)) >= 0.0:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            if a6da < -1.0 * da2:
                a4_4 = 3.0 * (a4_2 - a4_1)
                a4_3 = a4_2 - a4_4
            elif a6da > da2:
                a4_4 = 3.0 * (a4_3 - a4_1)
                a4_2 = a4_3 - a4_4
    # remap_constraint
    if iv >= 2:
        da1 = a4_3 - a4_2
        da2 = da1 * da1
        a6da = a4_4 * da1
        if extm == 1:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            if a6da < -da2:
                a4_4 = 3.0 * (a4_2 - a4_1)
                a4_3 = a4_2 - a4_4
            elif a6da > da2:
                a4_4 = 3.0 * (a4_3 - a4_1)
                a4_2 = a4_3 - a4_4
    return a4_1, a4_2, a4_3, a4_4


def set_vals(
    gam: FloatField,
    q: FloatField,
    delp: FloatField,
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    q_bot: FloatField,
    qs: FloatFieldIJ,
):
    from __externals__ import iv, kord

    with computation(PARALLEL), interval(0, 1):
        # set top
        if __INLINED(iv == -2):
            # gam = 0.5
            q = 1.5 * a4_1
        else:
            grid_ratio = delp[0, 0, 1] / delp
            bet = grid_ratio * (grid_ratio + 0.5)
            q = (
                (grid_ratio + grid_ratio) * (grid_ratio + 1.0) * a4_1 + a4_1[0, 0, 1]
            ) / bet
            gam = (1.0 + grid_ratio * (grid_ratio + 1.5)) / bet
    with computation(FORWARD):
        with interval(1, 2):
            if __INLINED(iv == -2):
                gam = 0.5
                grid_ratio = delp[0, 0, -1] / delp
                bet = 2.0 + grid_ratio + grid_ratio - gam
                q = (3.0 * (a4_1[0, 0, -1] + a4_1) - q[0, 0, -1]) / bet
        with interval(1, -1):
            if __INLINED(iv != -2):
                # set middle
                d4 = delp[0, 0, -1] / delp
                bet = 2.0 + d4 + d4 - gam[0, 0, -1]
                q = (3.0 * (a4_1[0, 0, -1] + d4 * a4_1) - q[0, 0, -1]) / bet
                gam = d4 / bet
        with interval(2, -2):
            if __INLINED(iv == -2):
                # set middle
                old_grid_ratio = delp[0, 0, -2] / delp[0, 0, -1]
                old_bet = 2.0 + old_grid_ratio + old_grid_ratio - gam[0, 0, -1]
                gam = old_grid_ratio / old_bet
                grid_ratio = delp[0, 0, -1] / delp
                bet = 2.0 + grid_ratio + grid_ratio - gam
                q = (3.0 * (a4_1[0, 0, -1] + a4_1) - q[0, 0, -1]) / bet
                # gam[0, 0, 1] = grid_ratio / bet
    with computation(FORWARD), interval(-2, -1):
        if __INLINED(iv == -2):
            # set bottom
            old_grid_ratio = delp[0, 0, -2] / delp[0, 0, -1]
            old_bet = 2.0 + old_grid_ratio + old_grid_ratio - gam[0, 0, -1]
            gam = old_grid_ratio / old_bet
            grid_ratio = delp[0, 0, -1] / delp
            q = (3.0 * (a4_1[0, 0, -1] + a4_1) - grid_ratio * qs - q[0, 0, -1]) / (
                2.0 + grid_ratio + grid_ratio - gam
            )
            q_bot = qs
    with computation(PARALLEL), interval(-1, None):
        if __INLINED(iv == -2):
            q_bot = qs
            q = qs
        else:
            # set bottom
            d4 = delp[0, 0, -2] / delp[0, 0, -1]
            a_bot = 1.0 + d4 * (d4 + 1.5)
            q = (
                2.0 * d4 * (d4 + 1.0) * a4_1[0, 0, -1]
                + a4_1[0, 0, -2]
                - a_bot * q[0, 0, -1]
            ) / (d4 * (d4 + 0.5) - a_bot * gam[0, 0, -1])
    with computation(BACKWARD), interval(0, -1):
        if __INLINED(iv != -2):
            q = q - gam * q[0, 0, 1]
    with computation(BACKWARD), interval(0, -2):
        if __INLINED(iv == -2):
            q = q - gam[0, 0, 1] * q[0, 0, 1]
    # set_avals
    with computation(PARALLEL):
        with interval(0, -1):
            if __INLINED(kord > 16):
                a4_2 = q
                a4_3 = q[0, 0, 1]
                a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
        with interval(-1, None):
            if __INLINED(kord > 16):
                a4_2 = q
                a4_3 = q_bot
                a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


def apply_constraints(
    q: FloatField,
    gam: FloatField,
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    ext5: FloatField,
    ext6: FloatField,
    extm: FloatField,
):
    from __externals__ import iv, kord

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
                if __INLINED(iv == 0):
                    q = 0.0 if (q < 0.0) else q
            # q = constrain_interior(q, gam, a4_1)
        with interval(-1, None):
            # do bottom
            q = q if q < tmp else tmp
            q = q if q > tmp2 else tmp2
    with computation(PARALLEL), interval(...):
        # re-set a4_2 and a4_3
        a4_2 = q
        a4_3 = q[0, 0, 1]
    # set_extm
    with computation(PARALLEL):
        with interval(0, 1):
            extm = (a4_2 - a4_1) * (a4_3 - a4_1) > 0.0
        with interval(1, -1):
            extm = gam * gam[0, 0, 1] < 0.0
        with interval(-1, None):
            extm = (a4_2 - a4_1) * (a4_3 - a4_1) > 0.0
    # set_exts
    with computation(PARALLEL), interval(...):
        if __INLINED(kord > 9):
            x0 = 2.0 * a4_1 - (a4_2 + a4_3)
            x1 = abs(a4_2 - a4_3)
            a4_4 = 3.0 * x0
            ext5 = abs(x0) > x1
            ext6 = abs(a4_4) > x1


def set_top(
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    extm: FloatField,
):
    from __externals__ import iv

    # set_top_as_iv0
    with computation(PARALLEL):
        with interval(0, 1):
            if __INLINED(iv == 0):
                a4_2 = a4_2 if a4_2 > 0.0 else 0.0
        with interval(...):
            if __INLINED(iv == 0):
                a4_4 = 3 * (2 * a4_1 - (a4_2 + a4_3))
    # set_top_as_iv1
    with computation(PARALLEL):
        with interval(0, 1):
            if __INLINED(iv == -1):
                a4_2 = 0.0 if a4_2 * a4_1 <= 0.0 else a4_2
        with interval(...):
            if __INLINED(iv == -1):
                a4_4 = 3 * (2 * a4_1 - (a4_2 + a4_3))
    # set_top_as_iv2
    with computation(PARALLEL):
        with interval(0, 1):
            if __INLINED(iv == 2):
                a4_2 = a4_1
                a4_3 = a4_1
                a4_4 = 0.0
        with interval(1, None):
            if __INLINED(iv == 2):
                a4_4 = 3 * (2 * a4_1 - (a4_2 + a4_3))
    # set_top_as_else
    with computation(PARALLEL):
        with interval(...):
            if __INLINED(iv < -1 or iv == 1 or iv > 2):
                a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
        with interval(0, -1):
            if __INLINED(iv != 2):
                a4_1, a4_2, a4_3, a4_4 = remap_constraint(
                    a4_1, a4_2, a4_3, a4_4, extm, 1
                )
        with interval(1, None):
            a4_1, a4_2, a4_3, a4_4 = remap_constraint(a4_1, a4_2, a4_3, a4_4, extm, 2)


def set_inner(
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    gam: FloatField,
    extm: FloatField,
    ext5: FloatField,
    ext6: FloatField,
    qmin: float,
):
    from __externals__ import iv, kord

    with computation(PARALLEL), interval(...):
        # set_inner_as_kordsmall
        if __INLINED(kord < 9):
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
        # set_inner_as_kord9
        if __INLINED(kord == 9):
            pmp_1 = a4_1 - 2.0 * gam[0, 0, 1]
            lac_1 = pmp_1 + 1.5 * gam[0, 0, 2]
            pmp_2 = a4_1 + 2.0 * gam
            lac_2 = pmp_2 - 1.5 * gam[0, 0, -1]
            tmp_min = a4_1
            tmp_max = a4_2
            tmp_max0 = a4_1
            if (
                (extm != 0.0 and extm[0, 0, -1] != 0.0)
                or (extm != 0.0 and extm[0, 0, 1] != 0.0)
                or (extm > 0.0 and (qmin > 0.0 and a4_1 < qmin))
            ):
                a4_2 = a4_1
                a4_3 = a4_1
                a4_4 = 0.0
            else:
                a4_4 = 6.0 * a4_1 - 3.0 * (a4_2 + a4_3)
                if abs(a4_4) > abs(a4_2 - a4_3):
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
        # set_inner_as_kord10
        if __INLINED(kord == 10):
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
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
        # remap_constraint
        if __INLINED(iv == 0):
            a4_1, a4_2, a4_3, a4_4 = remap_constraint(a4_1, a4_2, a4_3, a4_4, extm, 0)


def set_bottom(
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    extm: FloatField,
):
    from __externals__ import iv

    # set_bottom_as_iv0
    with computation(PARALLEL), interval(1, None):
        if __INLINED(iv == 0):
            a4_3 = a4_3 if a4_3 > 0.0 else 0.0
    # set_bottom_as_iv1
    with computation(PARALLEL), interval(-1, None):
        if __INLINED(iv == -1):
            a4_3 = 0.0 if a4_3 * a4_1 <= 0.0 else a4_3
    with computation(PARALLEL), interval(...):
        # set_bottom_as_iv0
        if __INLINED(iv == 0):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
        # set_bottom_as_iv1
        if __INLINED(iv == -1):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
        # set_bottom_as_else
        if __INLINED(iv > 0 or iv < -1):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
    with computation(PARALLEL), interval(0, -1):
        a4_1, a4_2, a4_3, a4_4 = remap_constraint(a4_1, a4_2, a4_3, a4_4, extm, 2)
    with computation(PARALLEL), interval(1, None):
        a4_1, a4_2, a4_3, a4_4 = remap_constraint(a4_1, a4_2, a4_3, a4_4, extm, 1)


class RemapProfile:
    """
    This corresponds to the cs_profile routine in FV3
    """

    def __init__(
        self,
        kord: int,
        iv: int,
        i1: int,
        i2: int,
        j1: int,
        j2: int,
    ):
        """
        The constraints on the spline are set by kord and iv.
        Arguments:
            kord: ???
            iv: ???
            i1: The first i-element to compute on
            i2: The last i-element to compute on
            j1: The first j-element to compute on
            j2: The last j-element to compute on
        """
        assert kord <= 10, f"kord {kord} not implemented."
        grid = spec.grid
        full_orig: Tuple[int] = grid.full_origin()
        km: int = grid.npz
        self._kord = kord

        self._gam: FloatField = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 0, 1)), origin=full_orig
        )
        self._q: FloatField = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 0, 1)), origin=full_orig
        )
        self._q_bot: FloatField = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 0, 1)), origin=full_orig
        )
        self._extm: FloatField = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 0, 1)), origin=full_orig
        )
        self._ext5: FloatField = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 0, 1)), origin=full_orig
        )
        self._ext6: FloatField = utils.make_storage_from_shape(
            grid.domain_shape_full(add=(0, 0, 1)), origin=full_orig
        )

        i_extent: int = i2 - i1 + 1
        j_extent: int = j2 - j1 + 1
        origin: Tuple[int, int, int] = (i1, j1, 0)
        domain: Tuple[int, int, int] = (i_extent, j_extent, km)
        domain_extend: Tuple[int, int, int] = (i_extent, j_extent, km + 1)

        self._set_values_stencil = FrozenStencil(
            func=set_vals,
            externals={"iv": iv, "kord": abs(kord)},
            origin=origin,
            domain=domain_extend,
        )

        self._apply_constraints_stencil = FrozenStencil(
            func=apply_constraints,
            externals={"iv": iv, "kord": abs(kord)},
            origin=origin,
            domain=domain,
        )

        self._set_top_stencil = FrozenStencil(
            func=set_top,
            externals={"iv": iv},
            origin=origin,
            domain=(i_extent, j_extent, 2),
        )

        self._set_set_inner_stencil = FrozenStencil(
            func=set_inner,
            externals={"iv": iv, "kord": abs(kord)},
            origin=(i1, j1, 2),
            domain=(i_extent, j_extent, km - 4),
        )

        self._set_bottom_stencil = FrozenStencil(
            func=set_bottom,
            externals={"iv": iv},
            origin=(i1, j1, km - 2),
            domain=(i_extent, j_extent, 2),
        )

    def __call__(
        self,
        qs: FloatFieldIJ,
        a4_1: FloatField,
        a4_2: FloatField,
        a4_3: FloatField,
        a4_4: FloatField,
        delp: FloatField,
        qmin: float = 0.0,
    ):
        """
        Calculates the interpolation coefficients for a cubic-spline which models the
        distribution of the remapped field within each deformed grid cell.
        The constraints on the spline are set by kord and iv.
        Arguments:
            qs: Bottom boundary condition
            a4_1: The first interpolation coefficient
            a4_2: The second interpolation coefficient
            a4_3: The third interpolation coefficient
            a4_4: The fourth interpolation coefficient
            delp: The pressure difference between grid levels
            qmin: The minimum value the field can take in a cell
        """
        self._set_values_stencil(
            self._gam,
            self._q,
            delp,
            a4_1,
            a4_2,
            a4_3,
            a4_4,
            self._q_bot,
            qs,
        )

        if abs(self._kord) <= 16:
            self._apply_constraints_stencil(
                self._q,
                self._gam,
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                self._ext5,
                self._ext6,
                self._extm,
            )

            self._set_top_stencil(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                self._extm,
            )

            self._set_set_inner_stencil(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                self._gam,
                self._extm,
                self._ext5,
                self._ext6,
                qmin,
            )

            self._set_bottom_stencil(
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                self._extm,
            )

        return a4_1, a4_2, a4_3, a4_4
