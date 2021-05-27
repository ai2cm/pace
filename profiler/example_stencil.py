# The below stencil is a "light" version of fv3core xppm - mainly
# demonstrating behaviors when many regions are used

# flak8/blake can't parse this file since it's really part of the codebase
# but an external example of how to use the sentic reproducer.
# This scripts requries a verion of GT4PY in the env to run
# type: ignore
import gt4py
from gt4py.gtscript import (
    __INLINED,
    IJ,
    PARALLEL,
    Field,
    computation,
    function,
    horizontal,
    interval,
    region,
    stencil,
)


FloatField = Field[float]
FloatFieldIJ = Field[IJ, float]

backend = "gtx86"
shape = (64, 64, 79)
origin = (3, 0, 0)
build_info = {}
nhalo = 3
iord = 0
externals = {
    "i_start": nhalo,
    "j_start": nhalo,
    "i_end": shape[0] - nhalo - 1,
    "j_end": shape[1] - nhalo - 1,
    "iord": iord,
    "mord": abs(iord),
    "xt_minmax": True,
    "region_weight": 0,
}


@function
def sign(a, b):
    asignb = abs(a)
    if b > 0:
        asignb = asignb
    else:
        asignb = -asignb
    return asignb


# volume-conserving cubic with 2nd drv=0 at end point:
# non-monotonic
c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0

# PPM volume mean form
p1 = 7.0 / 12.0
p2 = -1.0 / 12.0

yppm_s11 = 11.0 / 14.0
yppm_s14 = 4.0 / 7.0
yppm_s15 = 3.0 / 14.0


@function
def xt_dxa_edge_0_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa + dxa[-1, 0]) * q - dxa * q[-1, 0, 0]) / (dxa[-1, 0] + dxa)
        + ((2.0 * dxa[1, 0] + dxa[2, 0]) * q[1, 0, 0] - dxa[1, 0] * q[2, 0, 0])
        / (dxa[1, 0] + dxa[2, 0])
    )


@function
def xt_dxa_edge_1_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa[-1, 0] + dxa[-2, 0]) * q[-1, 0, 0] - dxa[-1, 0] * q[-2, 0, 0])
        / (dxa[-2, 0] + dxa[-1, 0])
        + ((2.0 * dxa + dxa[1, 0]) * q - dxa * q[1, 0, 0]) / (dxa + dxa[1, 0])
    )


@function
def xt_dxa_edge_0(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_0_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        maxq = max(max(max(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@function
def xt_dxa_edge_1(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_1_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        maxq = max(max(max(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@function
def west_edge_iord8plus_0(
    q: FloatField,
    dxa: FloatFieldIJ,
    dm: FloatField,
):
    bl = yppm_s14 * dm[-1, 0, 0] + yppm_s11 * (q[-1, 0, 0] - q)
    xt = xt_dxa_edge_0(q, dxa)
    br = xt - q
    return bl, br


@function
def west_edge_iord8plus_1(
    q: FloatField,
    dxa: FloatFieldIJ,
    dm: FloatField,
):
    xt = xt_dxa_edge_1(q, dxa)
    bl = xt - q
    xt = yppm_s15 * q + yppm_s11 * q[1, 0, 0] - yppm_s14 * dm[1, 0, 0]
    br = xt - q
    return bl, br


@function
def west_edge_iord8plus_2(
    q: FloatField,
    dm: FloatField,
    al: FloatField,
):
    xt = yppm_s15 * q[-1, 0, 0] + yppm_s11 * q - yppm_s14 * dm
    bl = xt - q
    br = al[1, 0, 0] - q
    return bl, br


@function
def east_edge_iord8plus_0(
    q: FloatField,
    dm: FloatField,
    al: FloatField,
):
    bl = al - q
    xt = yppm_s15 * q[1, 0, 0] + yppm_s11 * q + yppm_s14 * dm
    br = xt - q
    return bl, br


@function
def east_edge_iord8plus_1(
    q: FloatField,
    dxa: FloatFieldIJ,
    dm: FloatField,
):
    xt = yppm_s15 * q + yppm_s11 * q[-1, 0, 0] + yppm_s14 * dm[-1, 0, 0]
    bl = xt - q
    xt = xt_dxa_edge_0(q, dxa)
    br = xt - q
    return bl, br


@function
def east_edge_iord8plus_2(
    q: FloatField,
    dxa: FloatFieldIJ,
    dm: FloatField,
):
    xt = xt_dxa_edge_1(q, dxa)
    bl = xt - q
    br = yppm_s11 * (q[1, 0, 0] - q) - yppm_s14 * dm[1, 0, 0]
    return bl, br


@function
def compute_al(q: FloatField, dxa: FloatFieldIJ):
    """
    Interpolate q at interface.

    Inputs:
        q: Transported scalar
        dxa: dx on A-grid (?)

    Returns:
        Interpolated quantity
    """
    from __externals__ import i_end, i_start

    al = p1 * (q[-1, 0, 0] + q) + p2 * (q[-2, 0, 0] + q[1, 0, 0])

    with horizontal(region[i_start - 1, :], region[i_end, :]):
        al = c1 * q[-2, 0, 0] + c2 * q[-1, 0, 0] + c3 * q
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        al = 0.5 * (
            ((2.0 * dxa[-1, 0] + dxa[-2, 0]) * q[-1, 0, 0] - dxa[-1, 0] * q[-2, 0, 0])
            / (dxa[-2, 0] + dxa[-1, 0])
            + ((2.0 * dxa[0, 0] + dxa[1, 0]) * q[0, 0, 0] - dxa[0, 0] * q[1, 0, 0])
            / (dxa[0, 0] + dxa[1, 0])
        )
    with horizontal(region[i_start + 1, :], region[i_end + 2, :]):
        al = c3 * q[-1, 0, 0] + c2 * q[0, 0, 0] + c1 * q[1, 0, 0]

    return al


@function
def fx1_fn(courant, br, b0, bl):
    if courant > 0.0:
        ret = (1.0 - courant) * (br[-1, 0, 0] - courant * b0[-1, 0, 0])
    else:
        ret = (1.0 + courant) * (bl + courant * b0)
    return ret


@function
def get_tmp(bl, b0, br):

    smt5 = (3.0 * abs(b0)) < abs(bl - br)

    if smt5[-1, 0, 0] or smt5[0, 0, 0]:
        tmp = 1.0
    else:
        tmp = 0.0

    return tmp


@function
def final_flux(courant, q, fx1, tmp):
    return q[-1, 0, 0] + fx1 * tmp if courant > 0.0 else q + fx1 * tmp


@function
def get_flux(q: FloatField, courant: FloatField, al: FloatField):
    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[1, 0, 0] - q[0, 0, 0]
    b0 = bl + br

    tmp = get_tmp(bl, b0, br)
    fx1 = fx1_fn(courant, br, b0, bl)
    return final_flux(courant, q, fx1, tmp)  # noqa


@function
def get_flux_ord8plus(
    q: FloatField, courant: FloatField, bl: FloatField, br: FloatField
):
    b0 = bl + br
    fx1 = fx1_fn(courant, br, b0, bl)
    return final_flux(courant, q, fx1, 1.0)


@function
def dm_iord8plus(q: FloatField):
    xt = 0.25 * (q[1, 0, 0] - q[-1, 0, 0])
    dqr = max(max(q, q[-1, 0, 0]), q[1, 0, 0]) - q
    dql = q - min(min(q, q[-1, 0, 0]), q[1, 0, 0])
    return sign(min(min(abs(xt), dqr), dql), xt)


@function
def al_iord8plus(q: FloatField, dm: FloatField):
    return 0.5 * (q[-1, 0, 0] + q) + 1.0 / 3.0 * (dm[-1, 0, 0] - dm)


@function
def blbr_iord8(q: FloatField, al: FloatField, dm: FloatField):
    xt = 2.0 * dm
    bl = -1.0 * sign(min(abs(xt), abs(al - q)), xt)
    br = sign(min(abs(xt), abs(al[1, 0, 0] - q)), xt)
    return bl, br


@function
def yppm_pert_ppm_standard_constraint_fcn(
    a0: FloatField, al: FloatField, ar: FloatField
):
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


@function
def compute_blbr_ord8plus(q: FloatField, dxa: FloatFieldIJ):
    from __externals__ import i_end, i_start, region_weight

    dm = dm_iord8plus(q)
    al = al_iord8plus(q, dm)

    bl, br = blbr_iord8(q, al, dm)

    if __INLINED(region_weight > 0):
        with horizontal(region[i_start - 1, :]):
            bl, br = west_edge_iord8plus_0(q, dxa, dm)
            bl, br = yppm_pert_ppm_standard_constraint_fcn(q, bl, br)

    if __INLINED(region_weight > 1):
        with horizontal(region[i_start, :]):
            bl, br = west_edge_iord8plus_1(q, dxa, dm)
            bl, br = yppm_pert_ppm_standard_constraint_fcn(q, bl, br)

    if __INLINED(region_weight > 2):
        with horizontal(region[i_start + 1, :]):
            bl, br = west_edge_iord8plus_2(q, dm, al)
            bl, br = yppm_pert_ppm_standard_constraint_fcn(q, bl, br)

    if __INLINED(region_weight > 3):
        with horizontal(region[i_end - 1, :]):
            bl, br = east_edge_iord8plus_0(q, dm, al)
            bl, br = yppm_pert_ppm_standard_constraint_fcn(q, bl, br)

    if __INLINED(region_weight > 4):
        with horizontal(region[i_end, :]):
            bl, br = east_edge_iord8plus_1(q, dxa, dm)
            bl, br = yppm_pert_ppm_standard_constraint_fcn(q, bl, br)

    if __INLINED(region_weight > 5):
        with horizontal(region[i_end + 1, :]):
            bl, br = east_edge_iord8plus_2(q, dxa, dm)
            bl, br = yppm_pert_ppm_standard_constraint_fcn(q, bl, br)

    return bl, br


def compute_x_flux(
    q: FloatField,
    courant: FloatField,
    dxa: FloatFieldIJ,
    xflux: FloatField,
):

    with computation(PARALLEL), interval(...):
        bl, br = compute_blbr_ord8plus(q, dxa)
        xflux = get_flux_ord8plus(q, courant, bl, br)  # noqa


def run_xppm_like_stencil():
    # Build X variation of the comput_x_flux_stencils with different
    # region count
    compute_x_flux_stencils = {}
    for rweight in range(7):
        externals["region_weight"] = rweight
        compute_x_flux_stencils[rweight] = stencil(
            backend=backend,
            definition=compute_x_flux,
            externals=externals,
            rebuild=False,
        )

    # Data storage allocation
    storage = gt4py.storage.ones(
        backend, default_origin=origin, shape=shape, dtype=float
    )
    storage2 = gt4py.storage.ones(
        backend, default_origin=origin, shape=shape, dtype=float
    )
    storage3 = gt4py.storage.ones(
        backend, default_origin=origin, shape=shape, dtype=float
    )
    storage_2D = gt4py.storage.ones(
        backend,
        default_origin=(origin[0], origin[1]),
        shape=(shape[0], shape[1]),
        dtype=float,
        mask=(True, True, False),
    )

    # Fill storage with actual values
    for i in range(shape[0]):
        for j in range(shape[1]):
            storage[i, j, :] = i + j * shape[0]

    # Warm up
    for key, stencil_fn in compute_x_flux_stencils.items():
        stencil_fn(storage, storage2, storage_2D, storage3)

    # Perf marked run
    for rweight, stencil_fn in compute_x_flux_stencils.items():
        stencil_fn(storage, storage2, storage_2D, storage3)


if __name__ == "__main__":
    run_xppm_like_stencil()
