from typing import Optional

from gt4py import gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import sign
from fv3core.utils.typing import FloatField


input_vars = ["q", "c"]
inputs_params = ["jord", "ifirst", "ilast"]
output_vars = ["flux"]

# volume-conserving cubic with 2nd drv=0 at end point:
# non-monotonic
c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0

# PPM volume mean form
p1 = 7.0 / 12.0
p2 = -1.0 / 12.0

s11 = 11.0 / 14.0
s14 = 4.0 / 7.0
s15 = 3.0 / 14.0


@gtscript.function
def final_flux(courant, q, fx1, tmp):
    return q[0, -1, 0] + fx1 * tmp if courant > 0.0 else q + fx1 * tmp


@gtscript.function
def fx1_fn(courant, br, b0, bl):
    if courant > 0.0:
        ret = (1.0 - courant) * (br[0, -1, 0] - courant * b0[0, -1, 0])
    else:
        ret = (1.0 + courant) * (bl + courant * b0)
    return ret


@gtscript.function
def get_tmp(bl, b0, br):
    from __externals__ import mord

    if mord == 5:
        smt5 = bl * br < 0
    else:
        smt5 = (3.0 * abs(b0)) < abs(bl - br)

    if smt5[0, -1, 0] or smt5[0, 0, 0]:
        tmp = 1.0
    else:
        tmp = 0.0

    return tmp


@gtscript.function
def get_flux(q: FloatField, courant: FloatField, al: FloatField):
    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[0, 1, 0] - q[0, 0, 0]
    b0 = bl + br

    tmp = get_tmp(bl, b0, br)
    fx1 = fx1_fn(courant, br, b0, bl)
    return final_flux(courant, q, fx1, tmp)


@gtscript.function
def get_flux_ord8plus(
    q: FloatField, courant: FloatField, bl: FloatField, br: FloatField
):
    b0 = bl + br
    fx1 = fx1_fn(courant, br, b0, bl)
    return final_flux(courant, q, fx1, 1.0)


@gtscript.function
def dm_jord8plus(q: FloatField):
    xt = 0.25 * (q[0, 1, 0] - q[0, -1, 0])
    dqr = max(max(q, q[0, -1, 0]), q[0, 1, 0]) - q
    dql = q - min(min(q, q[0, -1, 0]), q[0, 1, 0])
    return sign(min(min(abs(xt), dqr), dql), xt)


@gtscript.function
def al_jord8plus(q: FloatField, dm: FloatField):
    return 0.5 * (q[0, -1, 0] + q) + 1.0 / 3.0 * (dm[0, -1, 0] - dm)


@gtscript.function
def blbr_jord8(q: FloatField, al: FloatField, dm: FloatField):
    xt = 2.0 * dm
    aldiff = al - q
    aldiffj = al[0, 1, 0] - q
    bl = -1.0 * sign(min(abs(xt), abs(aldiff)), xt)
    br = sign(min(abs(xt), abs(aldiffj)), xt)
    return bl, br


@gtscript.function
def xt_dya_edge_0_base(q, dya):
    return 0.5 * (
        ((2.0 * dya + dya[0, -1, 0]) * q - dya * q[0, -1, 0]) / (dya[0, -1, 0] + dya)
        + ((2.0 * dya[0, 1, 0] + dya[0, 2, 0]) * q[0, 1, 0] - dya[0, 1, 0] * q[0, 2, 0])
        / (dya[0, 1, 0] + dya[0, 2, 0])
    )


@gtscript.function
def xt_dya_edge_1_base(q, dya):
    return 0.5 * (
        (
            (2.0 * dya[0, -1, 0] + dya[0, -2, 0]) * q[0, -1, 0]
            - dya[0, -1, 0] * q[0, -2, 0]
        )
        / (dya[0, -2, 0] + dya[0, -1, 0])
        + ((2.0 * dya + dya[0, 1, 0]) * q - dya * q[0, 1, 0]) / (dya + dya[0, 1, 0])
    )


@gtscript.function
def xt_dya_edge_0(q, dya):
    from __externals__ import xt_minmax

    xt = xt_dya_edge_0_base(q, dya)
    if __INLINED(xt_minmax):
        minq = min(min(min(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        maxq = max(max(max(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dya_edge_1(q, dya):
    from __externals__ import xt_minmax

    xt = xt_dya_edge_1_base(q, dya)
    if __INLINED(xt_minmax):
        minq = min(min(min(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        maxq = max(max(max(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def south_edge_jord8plus_0(q: FloatField, dya: FloatField, dm: FloatField):
    bl = s14 * dm[0, -1, 0] + s11 * (q[0, -1, 0] - q)
    xt = xt_dya_edge_0(q, dya)
    br = xt - q
    return bl, br


@gtscript.function
def south_edge_jord8plus_1(q: FloatField, dya: FloatField, dm: FloatField):
    xt = xt_dya_edge_1(q, dya)
    bl = xt - q
    xt = s15 * q + s11 * q[0, 1, 0] - s14 * dm[0, 1, 0]
    br = xt - q
    return bl, br


@gtscript.function
def south_edge_jord8plus_2(q: FloatField, dm: FloatField, al: FloatField):
    xt = s15 * q[0, -1, 0] + s11 * q - s14 * dm
    bl = xt - q
    br = al[0, 1, 0] - q
    return bl, br


@gtscript.function
def north_edge_jord8plus_0(q: FloatField, dm: FloatField, al: FloatField):
    bl = al - q
    xt = s15 * q[0, 1, 0] + s11 * q + s14 * dm
    br = xt - q
    return bl, br


@gtscript.function
def north_edge_jord8plus_1(q: FloatField, dya: FloatField, dm: FloatField):
    xt = s15 * q + s11 * q[0, -1, 0] + s14 * dm[0, -1, 0]
    bl = xt - q
    xt = xt_dya_edge_0(q, dya)
    br = xt - q
    return bl, br


@gtscript.function
def north_edge_jord8plus_2(q: FloatField, dya: FloatField, dm: FloatField):
    xt = xt_dya_edge_1(q, dya)
    bl = xt - q
    br = s11 * (q[0, 1, 0] - q) - s14 * dm[0, 1, 0]
    return bl, br


@gtscript.function
def pert_ppm_positive_definite_constraint_fcn(
    a0: FloatField, al: FloatField, ar: FloatField
):
    da1 = 0.0
    a4 = 0.0
    fmin = 0.0
    if a0 <= 0.0:
        al = 0.0
        ar = 0.0
    else:
        a4 = -3.0 * (ar + al)
        da1 = ar - al
        if abs(da1) < -a4:
            fmin = a0 + 0.25 / a4 * da1 ** 2 + a4 * (1.0 / 12.0)
            if fmin < 0.0:
                if ar > 0.0 and al > 0.0:
                    ar = 0.0
                    al = 0.0
                elif da1 > 0.0:
                    ar = -2.0 * al
            else:
                al = -2.0 * ar

    return al, ar


@gtstencil()
def pert_ppm_positive_definite_constraint(
    a0: FloatField, al: FloatField, ar: FloatField
):
    with computation(PARALLEL), interval(...):
        al, ar = pert_ppm_positive_definite_constraint_fcn(a0, al, ar)


@gtscript.function
def pert_ppm_standard_constraint_fcn(a0: FloatField, al: FloatField, ar: FloatField):
    da1 = 0.0
    da2 = 0.0
    a6da = 0.0
    if al * ar < 0.0:
        da1 = al - ar
        da2 = da1 ** 2
        a6da = 3.0 * (al + ar) * da1
        if a6da < -da2:
            ar = -2.0 * al
        elif a6da > da2:
            al = -2.0 * ar
    else:
        # effect of dm=0 included here
        al = 0.0
        ar = 0.0
    return al, ar


@gtstencil()
def pert_ppm_standard_constraint(a0: FloatField, al: FloatField, ar: FloatField):
    with computation(PARALLEL), interval(...):
        al, ar = pert_ppm_standard_constraint_fcn(a0, al, ar)


@gtscript.function
def compute_al(q: FloatField, dya: FloatField):
    """
    Interpolate q at interface.

    Inputs:
        q: Transported scalar
        dya: dy on A-grid (?)

    Returns:
        Interpolated quantity
    """
    from __externals__ import j_end, j_start, jord

    assert __INLINED(jord < 8), "Not implemented"
    # {
    al = p1 * (q[0, -1, 0] + q) + p2 * (q[0, -2, 0] + q[0, 1, 0])

    if __INLINED(jord < 0):
        assert __INLINED(False), "Not tested"
        al = max(al, 0.0)

    with horizontal(region[:, j_start - 1], region[:, j_end]):
        al = c1 * q[0, -2, 0] + c2 * q[0, -1, 0] + c3 * q

    with horizontal(region[:, j_start], region[:, j_end + 1]):
        al = 0.5 * (
            (
                (2.0 * dya[0, -1, 0] + dya[0, -2, 0]) * q[0, -1, 0]
                - dya[0, -1, 0] * q[0, -2, 0]
            )
            / (dya[0, -2, 0] + dya[0, -1, 0])
            + (
                (2.0 * dya[0, 0, 0] + dya[0, 1, 0]) * q[0, 0, 0]
                - dya[0, 0, 0] * q[0, 1, 0]
            )
            / (dya[0, 0, 0] + dya[0, 1, 0])
        )

    with horizontal(region[:, j_start + 1], region[:, j_end + 2]):
        al = c3 * q[0, -1, 0] + c2 * q[0, 0, 0] + c1 * q[0, 1, 0]

    # }

    return al


@gtscript.function
def compute_blbr_ord8plus(q: FloatField, dya: FloatField):
    from __externals__ import j_end, j_start, jord, namelist

    bl = 0.0
    br = 0.0

    dm = dm_jord8plus(q)
    al = al_jord8plus(q, dm)

    assert __INLINED(jord == 8)
    # {
    bl, br = blbr_jord8(q, al, dm)
    # }

    assert __INLINED(namelist.grid_type < 3)
    # {
    with horizontal(region[:, j_start - 1]):
        bl, br = south_edge_jord8plus_0(q, dya, dm)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_start]):
        bl, br = south_edge_jord8plus_1(q, dya, dm)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_start + 1]):
        bl, br = south_edge_jord8plus_2(q, dm, al)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_end - 1]):
        bl, br = north_edge_jord8plus_0(q, dm, al)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_end]):
        bl, br = north_edge_jord8plus_1(q, dya, dm)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_end + 1]):
        bl, br = north_edge_jord8plus_2(q, dya, dm)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)
    # }

    return bl, br


# Optimized PPM in perturbation form:
def pert_ppm(a0, al, ar, iv, istart, jstart, kstart, ni, nj, nk):
    r12 = 1.0 / 12.0
    if iv == 0:
        pert_ppm_positive_definite_constraint(
            a0, al, ar, r12, origin=(istart, jstart, kstart), domain=(ni, nj, nk)
        )
    else:
        pert_ppm_standard_constraint(
            a0, al, ar, origin=(istart, jstart, kstart), domain=(ni, nj, nk)
        )
    return al, ar


def _compute_flux_stencil(
    q: FloatField, courant: FloatField, dya: FloatField, yflux: FloatField
):
    from __externals__ import mord

    with computation(PARALLEL), interval(...):
        if __INLINED(mord < 8):
            al = compute_al(q, dya)
            yflux = get_flux(q, courant, al)
        else:
            bl, br = compute_blbr_ord8plus(q, dya)
            yflux = get_flux_ord8plus(q, courant, bl, br)


def compute_flux(
    q: FloatField,
    c: FloatField,
    flux: FloatField,
    jord: int,
    ifirst: int,
    ilast: int,
    kstart: int = 0,
    nk: Optional[int] = None,
):
    """
    Compute y-flux using the PPM method.

    Args:
        q (in): Transported scalar
        c (in): Courant number
        flux (out): Flux
        jord: Method selector
        ifirst: Starting index of the I-dir compute domain
        ilast: Final index of the I-dir compute domain
        kstart: First index of the K-dir compute domain
        nk: Number of indices in the K-dir compute domain
    """
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    mord = abs(jord)
    if mord not in [5, 6, 7, 8]:
        raise NotImplementedError(
            f"Unimplemented hord value, {jord}. "
            "Currently only support hord={5, 6, 7, 8}"
        )

    stencil = gtstencil(
        definition=_compute_flux_stencil,
        externals={
            "jord": jord,
            "mord": mord,
            "xt_minmax": True,
        },
    )
    ni = ilast - ifirst + 1
    stencil(
        q,
        c,
        grid.dya,
        flux,
        origin=(ifirst, grid.js, kstart),
        domain=(ni, grid.njc + 1, nk),
    )
