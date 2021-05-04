import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.a2b_ord4 import a1, a2, lagrange_x_func, lagrange_y_func
from fv3core.utils import corners
from fv3core.utils.typing import FloatField, FloatFieldIJ


c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0
OFFSET = 2


def grid():
    return spec.grid


@gtstencil
def set_tmps(utmp: FloatField, vtmp: FloatField, big_number: float):
    with computation(PARALLEL), interval(...):
        utmp = big_number
        vtmp = big_number


@gtstencil
def fill_corners_x(utmp: FloatField, vtmp: FloatField, ua: FloatField, va: FloatField):
    with computation(PARALLEL), interval(...):
        utmp = corners.fill_corners_3cells_mult_x(
            utmp, vtmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        ua = corners.fill_corners_2cells_mult_x(
            ua, va, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )


@gtstencil
def fill_corners_y(utmp: FloatField, vtmp: FloatField, ua: FloatField, va: FloatField):
    with computation(PARALLEL), interval(...):
        vtmp = corners.fill_corners_3cells_mult_y(
            vtmp, utmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        va = corners.fill_corners_2cells_mult_y(
            va, ua, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )


@gtstencil
def east_west_edges(
    u: FloatField,
    ua: FloatField,
    uc: FloatField,
    utc: FloatField,
    utmp: FloatField,
    v: FloatField,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    cosa_u: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    dxa: FloatFieldIJ,
):
    from __externals__ import i_end, i_start, local_je, local_js

    with computation(PARALLEL), interval(...):
        # West
        with horizontal(region[i_start - 1, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x(utmp)

        faketmp = 0  # noqa
        with horizontal(region[i_start, local_js - 1 : local_je + 2]):
            utc = edge_interpolate4_x(ua, dxa)
            uc = utc * sin_sg3[-1, 0] if utc > 0 else utc * sin_sg1

        with horizontal(region[i_start + 1, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x_rev(utmp)

        with horizontal(region[i_start - 1, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        with horizontal(region[i_start + 1, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        # East
        with horizontal(region[i_end, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x(utmp)

        with horizontal(region[i_end + 1, local_js - 1 : local_je + 2]):
            utc = edge_interpolate4_x(ua, dxa)
            uc = utc * sin_sg3[-1, 0] if utc > 0 else utc * sin_sg1

        with horizontal(region[i_end + 2, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x_rev(utmp)

        with horizontal(region[i_end, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        with horizontal(region[i_end + 2, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)


@gtstencil
def north_south_edges(
    v: FloatField,
    va: FloatField,
    vc: FloatField,
    vtc: FloatField,
    vtmp: FloatField,
    u: FloatField,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    dya: FloatFieldIJ,
):
    from __externals__ import j_end, j_start, local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        faketmp = 0  # noqa
        with horizontal(
            region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 3]
        ):
            vc = lagrange_y_func(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_start - 1]):
            vc = vol_conserv_cubic_interp_func_y(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_start]):
            vtc = edge_interpolate4_y(va, dya)
            vc = vtc * sin_sg4[0, -1] if vtc > 0 else vtc * sin_sg2

        with horizontal(region[local_is - 1 : local_ie + 2, j_start + 1]):
            vc = vol_conserv_cubic_interp_func_y_rev(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_end]):
            vc = vol_conserv_cubic_interp_func_y(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_end + 1]):
            vtc = edge_interpolate4_y(va, dya)
            vc = vtc * sin_sg4[0, -1] if vtc > 0 else vtc * sin_sg2

        with horizontal(region[local_is - 1 : local_ie + 2, j_end + 2]):
            vc = vol_conserv_cubic_interp_func_y_rev(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)


# almost the same as a2b_ord4's version
@gtscript.function
def lagrange_y_func_p1(qx):
    return a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0])


@gtstencil
def lagrange_interpolation_y_p1(qx: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_y_func_p1(qx)


@gtscript.function
def lagrange_x_func_p1(qy):
    return a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0])


@gtstencil
def lagrange_interpolation_x_p1(qy: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_x_func_p1(qy)


@gtstencil
def avg_box(u: FloatField, v: FloatField, utmp: FloatField, vtmp: FloatField):
    with computation(PARALLEL), interval(...):
        utmp = 0.5 * (u + u[0, 1, 0])
        vtmp = 0.5 * (v + v[1, 0, 0])


@gtscript.function
def contravariant(v1, v2, cosa, rsin2):
    """
    Retrieve the contravariant component of the wind from its covariant
    component and the covariant component in the "other" (x/y) direction.

    For an orthogonal grid, cosa would be 0 and rsin2 would be 1, meaning
    the contravariant component is equal to the covariant component.
    However, the gnomonic cubed sphere grid is not orthogonal.

    Args:
        v1: covariant component of the wind for which we want to get the
            contravariant component
        v2: covariant component of the wind for the other direction,
            i.e. y if v1 is in x, x if v1 is in y
        cosa: cosine of the angle between the local x-direction and y-direction.
        rsin2: 1 / (sin(alpha))^2, where alpha is the angle between the local
            x-direction and y-direction

    Returns:
        v1_contravariant: contravariant component of v1
    """
    # From technical docs on FV3 cubed sphere grid:
    # The gnomonic cubed sphere grid is not orthogonal, meaning
    # the u and v vectors have some overlapping component. We can decompose
    # the total wind U in two ways, as a linear combination of the
    # coordinate vectors ("contravariant"):
    #    U = u_contravariant * u_dir + v_contravariant * v_dir
    # or as the projection of the vector onto the coordinate
    #    u_covariant = U dot u_dir
    #    v_covariant = U dot v_dir
    # The names come from the fact that the covariant vectors vary
    # (under a change in coordinate system) the same way the coordinate values do,
    # while the contravariant vectors vary in the "opposite" way.
    #
    # equations from FV3 technical documentation
    # u_cov = u_contra + v_contra * cos(alpha)  (eq 3.4)
    # v_cov = u_contra * cos(alpha) + v_contra  (eq 3.5)
    #
    # u_contra = u_cov - v_contra * cos(alpha)  (1, from 3.4)
    # v_contra = v_cov - u_contra * cos(alpha)  (2, from 3.5)
    # u_contra = u_cov - (v_cov - u_contra * cos(alpha)) * cos(alpha)  (from 1 & 2)
    # u_contra = u_cov - v_cov * cos(alpha) + u_contra * cos2(alpha) (follows)
    # u_contra * (1 - cos2(alpha)) = u_cov - v_cov * cos(alpha)
    # u_contra = u_cov/(1 - cos2(alpha)) - v_cov * cos(alpha)/(1 - cos2(alpha))
    # matches because rsin = 1 /(1 + cos2(alpha)),
    #                 cosa*rsin = cos(alpha)/(1 + cos2(alpha))

    # recall that:
    # rsin2 is 1/(sin(alpha))^2
    # cosa is cos(alpha)

    return (v1 - v2 * cosa) * rsin2


@gtstencil
def contravariant_stencil(
    u: FloatField,
    v: FloatField,
    cosa: FloatFieldIJ,
    rsin: FloatFieldIJ,
    out: FloatField,
):
    with computation(PARALLEL), interval(...):
        out = contravariant(u, v, cosa, rsin)


@gtstencil
def contravariant_components(
    utmp: FloatField,
    vtmp: FloatField,
    cosa_s: FloatFieldIJ,
    rsin2: FloatFieldIJ,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        ua = contravariant(utmp, vtmp, cosa_s, rsin2)
        va = contravariant(vtmp, utmp, cosa_s, rsin2)


@gtstencil
def ut_main(
    utmp: FloatField,
    uc: FloatField,
    v: FloatField,
    cosa_u: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    ut: FloatField,
):
    with computation(PARALLEL), interval(...):
        uc = lagrange_x_func(utmp)
        ut = contravariant(uc, v, cosa_u, rsin_u)


@gtstencil
def vt_main(
    vtmp: FloatField,
    vc: FloatField,
    u: FloatField,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    vt: FloatField,
):
    with computation(PARALLEL), interval(...):
        vc = lagrange_y_func(vtmp)
        vt = contravariant(vc, u, cosa_v, rsin_v)


@gtscript.function
def vol_conserv_cubic_interp_func_x(u):
    return c1 * u[-2, 0, 0] + c2 * u[-1, 0, 0] + c3 * u


@gtscript.function
def vol_conserv_cubic_interp_func_x_rev(u):
    return c1 * u[1, 0, 0] + c2 * u + c3 * u[-1, 0, 0]


@gtscript.function
def vol_conserv_cubic_interp_func_y(v):
    return c1 * v[0, -2, 0] + c2 * v[0, -1, 0] + c3 * v


@gtscript.function
def vol_conserv_cubic_interp_func_y_rev(v):
    return c1 * v[0, 1, 0] + c2 * v + c3 * v[0, -1, 0]


@gtstencil
def vol_conserv_cubic_interp_x(utmp: FloatField, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc = vol_conserv_cubic_interp_func_x(utmp)


@gtstencil
def vol_conserv_cubic_interp_x_rev(utmp: FloatField, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc = vol_conserv_cubic_interp_func_x_rev(utmp)


@gtstencil
def vol_conserv_cubic_interp_y(vtmp: FloatField, vc: FloatField):
    with computation(PARALLEL), interval(...):
        vc = vol_conserv_cubic_interp_func_y(vtmp)


@gtstencil
def vt_edge(
    vtmp: FloatField,
    vc: FloatField,
    u: FloatField,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    vt: FloatField,
    rev: int,
):
    with computation(PARALLEL), interval(...):
        vc = (
            vol_conserv_cubic_interp_func_y(vtmp)
            if rev == 0
            else vol_conserv_cubic_interp_func_y_rev(vtmp)
        )
        vt = contravariant(vc, u, cosa_v, rsin_v)


@gtstencil
def uc_x_edge1(
    ut: FloatField, sin_sg3: FloatFieldIJ, sin_sg1: FloatFieldIJ, uc: FloatField
):
    with computation(PARALLEL), interval(...):
        uc = ut * sin_sg3[-1, 0] if ut > 0 else ut * sin_sg1


@gtstencil
def vc_y_edge1(
    vt: FloatField, sin_sg4: FloatFieldIJ, sin_sg2: FloatFieldIJ, vc: FloatField
):
    with computation(PARALLEL), interval(...):
        vc = vt * sin_sg4[0, -1] if vt > 0 else vt * sin_sg2


@gtscript.function
def edge_interpolate4_x(ua, dxa):
    t1 = dxa[-2, 0] + dxa[-1, 0]
    t2 = dxa[0, 0] + dxa[1, 0]
    n1 = (t1 + dxa[-1, 0]) * ua[-1, 0, 0] - dxa[-1, 0] * ua[-2, 0, 0]
    n2 = (t1 + dxa[0, 0]) * ua[0, 0, 0] - dxa[0, 0] * ua[1, 0, 0]
    return 0.5 * (n1 / t1 + n2 / t2)


@gtscript.function
def edge_interpolate4_y(va, dya):
    t1 = dya[0, -2] + dya[0, -1]
    t2 = dya[0, 0] + dya[0, 1]
    n1 = (t1 + dya[0, -1]) * va[0, -1, 0] - dya[0, -1] * va[0, -2, 0]
    n2 = (t1 + dya[0, 0]) * va[0, 0, 0] - dya[0, 0] * va[0, 1, 0]
    return 0.5 * (n1 / t1 + n2 / t2)


def compute(dord4, uc, vc, u, v, ua, va, utc, vtc):
    if spec.namelist.grid_type >= 3:
        raise Exception("unimplemented grid_type >= 3")
    grid = spec.grid
    big_number = 1e30  # 1e8 if 32 bit
    nx = grid.ie + 1  # grid.npx + 2
    ny = grid.je + 1  # grid.npy + 2
    i1 = grid.is_ - 1
    j1 = grid.js - 1
    id_ = 1 if dord4 else 0
    pad = 2 + 2 * id_
    npt = 4 if not grid.nested else 0
    if npt > grid.nic - 1 or npt > grid.njc - 1:
        npt = 0
    utmp = utils.make_storage_from_shape(
        ua.shape, grid.full_origin(), cache_key="d2a2c_vect_utmp"
    )
    vtmp = utils.make_storage_from_shape(
        va.shape, grid.full_origin(), cache_key="d2a2c_vect_vtmp"
    )

    js1 = npt + OFFSET if grid.south_edge else grid.js - 1
    je1 = ny - npt if grid.north_edge else grid.je + 1
    is1 = npt + OFFSET if grid.west_edge else grid.isd
    ie1 = nx - npt if grid.east_edge else grid.ied

    is2 = npt + OFFSET if grid.west_edge else grid.is_ - 1
    ie2 = nx - npt if grid.east_edge else grid.ie + 1
    js2 = npt + OFFSET if grid.south_edge else grid.jsd
    je2 = ny - npt if grid.north_edge else grid.jed

    ifirst = grid.is_ + 2 if grid.west_edge else grid.is_ - 1
    ilast = grid.ie - 1 if grid.east_edge else grid.ie + 2
    idiff = ilast - ifirst + 1

    jfirst = grid.js + 2 if grid.south_edge else grid.js - 1
    jlast = grid.je - 1 if grid.north_edge else grid.je + 2
    jdiff = jlast - jfirst + 1

    js3 = npt + OFFSET if grid.south_edge else grid.jsd
    je3 = ny - npt if grid.north_edge else grid.jed
    jdiff3 = je3 - js3 + 1

    set_tmps(
        utmp,
        vtmp,
        big_number,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )

    lagrange_interpolation_y_p1(
        u, utmp, origin=(is1, js1, 0), domain=(ie1 - is1 + 1, je1 - js1 + 1, grid.npz)
    )

    lagrange_interpolation_x_p1(
        v, vtmp, origin=(is2, js2, 0), domain=(ie2 - is2 + 1, je2 - js2 + 1, grid.npz)
    )

    # tmp edges
    if grid.south_edge:
        avg_box(
            u,
            v,
            utmp,
            vtmp,
            origin=(grid.isd, grid.jsd, 0),
            domain=(grid.nid, npt + OFFSET - grid.jsd, grid.npz),
        )
    if grid.north_edge:
        je2 = ny - npt + 1
        avg_box(
            u,
            v,
            utmp,
            vtmp,
            origin=(grid.isd, je2, 0),
            domain=(grid.nid, grid.jed - je2 + 1, grid.npz),
        )

    if grid.west_edge:
        avg_box(
            u,
            v,
            utmp,
            vtmp,
            origin=(grid.isd, js3, 0),
            domain=(npt + OFFSET - grid.isd, jdiff3, grid.npz),
        )
    if grid.east_edge:
        avg_box(
            u,
            v,
            utmp,
            vtmp,
            origin=(nx + 1 - npt, js3, 0),
            domain=(grid.ied - nx + npt, jdiff3, grid.npz),
        )

    # contra-variant components at cell center
    contravariant_components(
        utmp,
        vtmp,
        grid.cosa_s,
        grid.rsin2,
        ua,
        va,
        origin=(grid.is_ - 1 - id_, grid.js - 1 - id_, 0),
        domain=(grid.nic + pad, grid.njc + pad, grid.npz),
    )
    # Fix the edges
    # Xdir:
    fill_corners_x(
        utmp,
        vtmp,
        ua,
        va,
        origin=grid.compute_origin(add=(-3, -3, 0)),
        domain=grid.domain_shape_compute(add=(6, 6, 0)),
    )

    ut_main(
        utmp,
        uc,
        v,
        grid.cosa_u,
        grid.rsin_u,
        utc,
        origin=(ifirst, j1, 0),
        domain=(idiff, grid.njc + 2, grid.npz),
    )

    east_west_edges(
        u,
        ua,
        uc,
        utc,
        utmp,
        v,
        grid.sin_sg1,
        grid.sin_sg3,
        grid.cosa_u,
        grid.rsin_u,
        grid.dxa,
        origin=grid.compute_origin(add=(-3, -3, 0)),
        domain=grid.domain_shape_compute(add=(6, 6, 0)),
    )

    # Ydir:
    fill_corners_y(
        utmp,
        vtmp,
        ua,
        va,
        origin=grid.compute_origin(add=(-3, -3, 0)),
        domain=grid.domain_shape_compute(add=(6, 6, 0)),
    )

    north_south_edges(
        v,
        va,
        vc,
        vtc,
        vtmp,
        u,
        grid.sin_sg2,
        grid.sin_sg4,
        grid.cosa_v,
        grid.rsin_v,
        grid.dya,
        origin=grid.compute_origin(add=(-3, -3, 0)),
        domain=grid.domain_shape_compute(add=(6, 6, 0)),
    )

    vt_main(
        vtmp,
        vc,
        u,
        grid.cosa_v,
        grid.rsin_v,
        vtc,
        origin=(i1, jfirst, 0),
        domain=(grid.nic + 2, jdiff, grid.npz),
    )
