from typing import Tuple

import gt4py.gtscript as gtscript
from gt4py.gtscript import __INLINED, BACKWARD, FORWARD, PARALLEL, computation, interval

import pace.dsl.gt4py_utils as utils
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import BoolField, FloatField, FloatFieldIJ


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
def posdef_constraint_iv0(
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
):
    """
    Args:
        a4_1 (inout): First cubic interpolation coefficient
        a4_2 (inout): Second cubic interpolation coefficient
        a4_3 (inout): Third cubic interpolation coefficient
        a4_4 (inout): Fourth cubic interpolation coefficient
    """
    if a4_1 <= 0.0:
        a4_2 = a4_1
        a4_3 = a4_1
        a4_4 = 0.0
    else:
        if (
            abs(a4_3 - a4_2) < -a4_4
            and (a4_1 + 0.25 * (a4_3 - a4_2) ** 2 / a4_4 + a4_4 * (1.0 / 12.0)) < 0.0
        ):
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
    return a4_1, a4_2, a4_3, a4_4


@gtscript.function
def posdef_constraint_iv1(
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
):
    """
    Args:
        a4_1 (inout): First cubic interpolation coefficient
        a4_2 (inout): Second cubic interpolation coefficient
        a4_3 (inout): Third cubic interpolation coefficient
        a4_4 (inout): Fourth cubic interpolation coefficient
    """
    da1 = a4_3 - a4_2
    da2 = da1 * da1
    a6da = a4_4 * da1
    if ((a4_1 - a4_2) * (a4_1 - a4_3)) >= 0.0:
        a4_2 = a4_1
        a4_3 = a4_1
        a4_4 = 0.0
    elif a6da < -1.0 * da2:
        a4_4 = 3.0 * (a4_2 - a4_1)
        a4_3 = a4_2 - a4_4
    elif a6da > da2:
        a4_4 = 3.0 * (a4_3 - a4_1)
        a4_2 = a4_3 - a4_4
    return a4_1, a4_2, a4_3, a4_4


@gtscript.function
def remap_constraint(
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    extm: BoolField,
):
    """
    Args:
        a4_1 (inout): First cubic interpolation coefficient
        a4_2 (inout): Second cubic interpolation coefficient
        a4_3 (inout): Third cubic interpolation coefficient
        a4_4 (inout): Fourth cubic interpolation coefficient
        extm (in): If true sets a4_2 and a4_3 as a4_1 and a4_4 to 0
    """
    da1 = a4_3 - a4_2
    da2 = da1 * da1
    a6da = a4_4 * da1
    if extm:
        a4_2 = a4_1
        a4_3 = a4_1
        a4_4 = 0.0
    elif a6da < -da2:
        a4_4 = 3.0 * (a4_2 - a4_1)
        a4_3 = a4_2 - a4_4
    elif a6da > da2:
        a4_4 = 3.0 * (a4_3 - a4_1)
        a4_2 = a4_3 - a4_4
    return a4_1, a4_2, a4_3, a4_4


def set_initial_vals(
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
    """
    Args:
        gam (out):
        q (out):
        delp (in):
        a4_1 (in):
        a4_2 (out):
        a4_3 (out):
        a4_4 (out):
        q_bot (out):
        qs (in):
    """
    from __externals__ import iv, kord

    with computation(FORWARD):
        with interval(0, 1):
            # set top
            if __INLINED(iv == -2):
                # gam = 0.5
                q = 1.5 * a4_1
            else:
                grid_ratio = delp[0, 0, 1] / delp
                bet = grid_ratio * (grid_ratio + 0.5)
                q = (
                    (grid_ratio + grid_ratio) * (grid_ratio + 1.0) * a4_1
                    + a4_1[0, 0, 1]
                ) / bet
                gam = (1.0 + grid_ratio * (grid_ratio + 1.5)) / bet
        with interval(1, 2):
            if __INLINED(iv == -2):
                gam = 0.5
                grid_ratio = delp[0, 0, -1] / delp
                bet = 2.0 + grid_ratio + grid_ratio - gam
                q = (3.0 * (a4_1[0, 0, -1] + a4_1) - q[0, 0, -1]) / bet
    with computation(FORWARD), interval(1, -1):
        if __INLINED(iv != -2):
            # set middle
            d4 = delp[0, 0, -1] / delp
            bet = 2.0 + d4 + d4 - gam[0, 0, -1]
            q = (3.0 * (a4_1[0, 0, -1] + d4 * a4_1) - q[0, 0, -1]) / bet
            gam = d4 / bet
    with computation(FORWARD):
        with interval(2, -1):
            if __INLINED(iv == -2):
                old_grid_ratio = delp[0, 0, -2] / delp[0, 0, -1]
                old_bet = 2.0 + old_grid_ratio + old_grid_ratio - gam[0, 0, -1]
                gam = old_grid_ratio / old_bet
                grid_ratio = delp[0, 0, -1] / delp
    with computation(FORWARD):
        with interval(2, -2):
            if __INLINED(iv == -2):
                # set middle
                bet = 2.0 + grid_ratio + grid_ratio - gam
                q = (3.0 * (a4_1[0, 0, -1] + a4_1) - q[0, 0, -1]) / bet
                # gam[0, 0, 1] = grid_ratio / bet
        with interval(-2, -1):
            if __INLINED(iv == -2):
                # set bottom
                q = (3.0 * (a4_1[0, 0, -1] + a4_1) - grid_ratio * qs - q[0, 0, -1]) / (
                    2.0 + grid_ratio + grid_ratio - gam
                )
                q_bot = qs
        with interval(-1, None):
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
    with computation(PARALLEL), interval(...):
        if __INLINED(kord > 16):
            a4_2 = q
            a4_3 = q[0, 0, 1]
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
    with computation(PARALLEL), interval(-1, None):
        if __INLINED(kord > 16):
            a4_3 = q_bot


def apply_constraints(
    q: FloatField,
    gam: FloatField,
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    ext5: BoolField,
    ext6: BoolField,
    extm: BoolField,
):
    """
    Args:
        q (inout):
        gam (out):
        a4_1 (in):
        a4_2 (out):
        a4_3 (out):
        a4_4 (out):
        ext5 (out):
        ext6 (out):
        extm (out):
    """
    # TODO: q is not consumed in remap_profile after this stencil.
    # Can we remove it or set it to be purely an input here?
    from __externals__ import iv, kord

    # apply constraints
    with computation(PARALLEL), interval(1, None):
        a4_1_0 = a4_1[0, 0, -1]
        tmp = a4_1_0 if a4_1_0 > a4_1 else a4_1
        tmp2 = a4_1_0 if a4_1_0 < a4_1 else a4_1
        gam = a4_1 - a4_1_0
    with computation(PARALLEL), interval(1, 2):
        # do top
        if q >= tmp:
            q = tmp
        if q <= tmp2:
            q = tmp2
    with computation(FORWARD):
        with interval(2, -1):
            # do middle
            if (gam[0, 0, -1] * gam[0, 0, 1]) > 0:
                if q >= tmp:
                    q = tmp
                if q <= tmp2:
                    q = tmp2
            elif gam[0, 0, -1] > 0:
                # there's a local maximum
                if q <= tmp2:
                    q = tmp2
            else:
                # there's a local minimum
                if q >= tmp:
                    q = tmp
                if __INLINED(iv == 0):
                    if q < 0.0:
                        q = 0.0
            # q = constrain_interior(q, gam, a4_1)
        with interval(-1, None):
            # do bottom
            if q >= tmp:
                q = tmp
            if q <= tmp2:
                q = tmp2
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


def set_interpolation_coefficients(
    gam: FloatField,
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    ext5: BoolField,
    ext6: BoolField,
    extm: BoolField,
    qmin: float,
):
    """
    Args:
        gam (in):
        a4_1 (inout):
        a4_2 (inout):
        a4_3 (inout):
        a4_4 (inout):
        ext5 (in):
        ext6 (in):
        extm (in):
        qmin (in):
    """
    from __externals__ import iv, kord

    # set_top_as_iv0
    with computation(PARALLEL), interval(0, 1):
        if __INLINED(iv == 0):
            if a4_2 < 0.0:
                a4_2 = 0.0
    with computation(PARALLEL), interval(0, 2):
        if __INLINED(iv == 0):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
    # set_top_as_iv1
    with computation(PARALLEL), interval(0, 1):
        if __INLINED(iv == -1):
            if a4_2 * a4_1 <= 0.0:
                a4_2 = 0.0
    with computation(PARALLEL), interval(0, 2):
        if __INLINED(iv == -1):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
    # set_top_as_iv2
    with computation(PARALLEL):
        with interval(0, 1):
            if __INLINED(iv == 2):
                a4_2 = a4_1
                a4_3 = a4_1
                a4_4 = 0.0
        with interval(1, 2):
            if __INLINED(iv == 2):
                a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
    # set_top_as_else
    with computation(PARALLEL), interval(0, 2):
        if __INLINED(iv < -1 or iv == 1 or iv > 2):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
    with computation(PARALLEL):
        with interval(0, 1):
            if __INLINED(iv != 2):
                a4_1, a4_2, a4_3, a4_4 = posdef_constraint_iv1(a4_1, a4_2, a4_3, a4_4)
        with interval(1, 2):
            a4_1, a4_2, a4_3, a4_4 = remap_constraint(a4_1, a4_2, a4_3, a4_4, extm)

    with computation(PARALLEL), interval(2, -2):
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
                (extm and extm[0, 0, -1])
                or (extm and extm[0, 0, 1])
                or (extm and (qmin > 0.0 and a4_1 < qmin))
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
            a4_1, a4_2, a4_3, a4_4 = posdef_constraint_iv0(a4_1, a4_2, a4_3, a4_4)
    # set_bottom_as_iv0, set_bottom_as_iv1
    # TODO(rheag) temporary workaround to gt:gpu bug
    # this computation can get out of order with the one that follows
    with computation(FORWARD), interval(-1, None):
        if __INLINED(iv == 0):
            if a4_3 < 0.0:
                a4_3 = 0.0
        if __INLINED(iv == -1):
            if a4_3 * a4_1 <= 0.0:
                a4_3 = 0.0
    with computation(PARALLEL), interval(-2, None):
        # set_bottom_as_iv0, set_bottom_as_iv1, set_bottom_as_else
        a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))
    with computation(FORWARD):
        with interval(-2, -1):
            a4_1, a4_2, a4_3, a4_4 = remap_constraint(a4_1, a4_2, a4_3, a4_4, extm)
        with interval(-1, None):
            a4_1, a4_2, a4_3, a4_4 = posdef_constraint_iv1(a4_1, a4_2, a4_3, a4_4)


class RemapProfile:
    """
    This corresponds to the cs_profile routine in FV3
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
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
            stencil_factory
            kord: ???
            iv: ???
            i1: The first i-element to compute on
            i2: The last i-element to compute on
            j1: The first j-element to compute on
            j2: The last j-element to compute on
        """
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )

        assert kord <= 10, f"kord {kord} not implemented."
        grid_indexing = stencil_factory.grid_indexing
        km: int = grid_indexing.domain[2]
        self._kord = kord

        def make_storage(**kwargs):
            return utils.make_storage_from_shape(
                shape=grid_indexing.domain_full(add=(0, 0, 1)),
                origin=grid_indexing.origin_full(),
                backend=stencil_factory.backend,
                **kwargs,
            )

        self._gam: FloatField = make_storage()
        self._q: FloatField = make_storage()
        self._q_bot: FloatField = make_storage()
        self._extm: BoolField = make_storage(dtype=bool)
        self._ext5: BoolField = make_storage()
        self._ext6: BoolField = make_storage()

        i_extent: int = int(i2 - i1 + 1)
        j_extent: int = int(j2 - j1 + 1)
        origin: Tuple[int, int, int] = (i1, j1, 0)
        domain: Tuple[int, int, int] = (i_extent, j_extent, km)
        domain_extend: Tuple[int, int, int] = (i_extent, j_extent, km + 1)

        self._set_initial_values = stencil_factory.from_origin_domain(
            func=set_initial_vals,
            externals={"iv": iv, "kord": abs(kord)},
            origin=origin,
            domain=domain_extend,
        )

        self._apply_constraints = stencil_factory.from_origin_domain(
            func=apply_constraints,
            externals={"iv": iv, "kord": abs(kord)},
            origin=origin,
            domain=domain,
        )

        self._set_interpolation_coefficients = stencil_factory.from_origin_domain(
            func=set_interpolation_coefficients,
            externals={"iv": iv, "kord": abs(kord)},
            origin=origin,
            domain=domain,
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
            qs (in): Bottom boundary condition
            a4_1 (out): The first interpolation coefficient
            a4_2 (out): The second interpolation coefficient
            a4_3 (out): The third interpolation coefficient
            a4_4 (out): The fourth interpolation coefficient
            delp (in): The pressure difference between grid levels
            qmin (in): The minimum value the field can take in a cell
        """
        self._set_initial_values(
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
            # TODO: These stencils could be combined once the backend can handle it
            self._apply_constraints(
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

            self._set_interpolation_coefficients(
                self._gam,
                a4_1,
                a4_2,
                a4_3,
                a4_4,
                self._ext5,
                self._ext6,
                self._extm,
                qmin,
            )
