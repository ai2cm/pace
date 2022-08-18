import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import pace.dsl.gt4py_utils as utils
from fv3core.stencils.a2b_ord4 import a1, a2, lagrange_x_func, lagrange_y_func
from pace.dsl.dace.orchestration import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.stencils import corners
from pace.util import X_DIM, Y_DIM, Z_DIM
from pace.util.grid import GridData


c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0
OFFSET = 2


def set_tmps(utmp: FloatField, vtmp: FloatField, big_number: float):
    with computation(PARALLEL), interval(...):
        utmp = big_number
        vtmp = big_number


# almost the same as a2b_ord4's version
@gtscript.function
def lagrange_y_func_p1(qx):
    return a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0])


def lagrange_interpolation_y_p1(qx: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_y_func_p1(qx)


@gtscript.function
def lagrange_x_func_p1(qy):
    return a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0])


def lagrange_interpolation_x_p1(qy: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_x_func_p1(qy)


def avg_box(u: FloatField, v: FloatField, utmp: FloatField, vtmp: FloatField):
    """
    D2A2C_AVG_OFFSET is an external that describes how far the
    averaging should go before switching to Lagrangian interpolation. For
    sufficiently small grids, this should be set to -1, otherwise 3. Note that
    this makes the stencil code in d2a2c grid-dependent!
    """
    from __externals__ import D2A2C_AVG_OFFSET, i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(
            region[:, : j_start + D2A2C_AVG_OFFSET],
            region[:, j_end - D2A2C_AVG_OFFSET + 1 :],
            region[: i_start + D2A2C_AVG_OFFSET, :],
            region[i_end - D2A2C_AVG_OFFSET + 1 :, :],
        ):
            utmp = 0.5 * (u + u[0, 1, 0])
            vtmp = 0.5 * (v + v[1, 0, 0])


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


def fill_corners_x(utmp: FloatField, vtmp: FloatField, ua: FloatField, va: FloatField):
    with computation(PARALLEL), interval(...):
        utmp = corners.fill_corners_3cells_mult_x(
            utmp, vtmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        ua = corners.fill_corners_2cells_mult_x(
            ua, va, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )


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


def fill_corners_y(utmp: FloatField, vtmp: FloatField, ua: FloatField, va: FloatField):
    with computation(PARALLEL), interval(...):
        vtmp = corners.fill_corners_3cells_mult_y(
            vtmp, utmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        va = corners.fill_corners_2cells_mult_y(
            va, ua, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )


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


def contravariant_stencil(
    u: FloatField,
    v: FloatField,
    cosa: FloatFieldIJ,
    rsin: FloatFieldIJ,
    out: FloatField,
):
    with computation(PARALLEL), interval(...):
        out = contravariant(u, v, cosa, rsin)


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


def vol_conserv_cubic_interp_x(utmp: FloatField, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc = vol_conserv_cubic_interp_func_x(utmp)


def vol_conserv_cubic_interp_x_rev(utmp: FloatField, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc = vol_conserv_cubic_interp_func_x_rev(utmp)


def vol_conserv_cubic_interp_y(vtmp: FloatField, vc: FloatField):
    with computation(PARALLEL), interval(...):
        vc = vol_conserv_cubic_interp_func_y(vtmp)


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


def uc_x_edge1(
    ut: FloatField, sin_sg3: FloatFieldIJ, sin_sg1: FloatFieldIJ, uc: FloatField
):
    with computation(PARALLEL), interval(...):
        uc = ut * sin_sg3[-1, 0] if ut > 0 else ut * sin_sg1


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


class DGrid2AGrid2CGridVectors:
    """
    Fortran name d2a2c_vect
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        nested: bool,
        grid_type: int,
        dord4: bool,
    ):
        orchestrate(self, config=stencil_factory.config.dace_config)

        grid_indexing = stencil_factory.grid_indexing
        self._cosa_s = grid_data.cosa_s
        self._cosa_u = grid_data.cosa_u
        self._cosa_v = grid_data.cosa_v
        self._rsin_u = grid_data.rsin_u
        self._rsin_v = grid_data.rsin_v
        self._rsin2 = grid_data.rsin2
        self._dxa = grid_data.dxa
        self._dya = grid_data.dya
        self._sin_sg1 = grid_data.sin_sg1
        self._sin_sg2 = grid_data.sin_sg2
        self._sin_sg3 = grid_data.sin_sg3
        self._sin_sg4 = grid_data.sin_sg4

        if grid_type >= 3:
            raise NotImplementedError("unimplemented grid_type >= 3")
        self._big_number = 1e30  # 1e8 if 32 bit
        nx = grid_indexing.iec + 1  # grid.npx + 2
        ny = grid_indexing.jec + 1  # grid.npy + 2
        i1 = grid_indexing.isc - 1
        j1 = grid_indexing.jsc - 1
        id_ = 1 if dord4 else 0
        pad = 2 + 2 * id_
        npt = 4 if not nested else 0
        if npt > grid_indexing.domain[0] - 1 or npt > grid_indexing.domain[1] - 1:
            npt = 0
        self._utmp = utils.make_storage_from_shape(
            grid_indexing.max_shape,
            grid_indexing.origin_full(),
            backend=stencil_factory.backend,
        )
        self._vtmp = utils.make_storage_from_shape(
            grid_indexing.max_shape,
            grid_indexing.origin_full(),
            backend=stencil_factory.backend,
        )

        js1 = npt + OFFSET if grid_indexing.south_edge else grid_indexing.jsc - 1
        je1 = ny - npt if grid_indexing.north_edge else grid_indexing.jec + 1
        is1 = npt + OFFSET if grid_indexing.west_edge else grid_indexing.isd
        ie1 = nx - npt if grid_indexing.east_edge else grid_indexing.ied

        is2 = npt + OFFSET if grid_indexing.west_edge else grid_indexing.isc - 1
        ie2 = nx - npt if grid_indexing.east_edge else grid_indexing.iec + 1
        js2 = npt + OFFSET if grid_indexing.south_edge else grid_indexing.jsd
        je2 = ny - npt if grid_indexing.north_edge else grid_indexing.jed

        ifirst = (
            grid_indexing.isc + 2 if grid_indexing.west_edge else grid_indexing.isc - 1
        )
        ilast = (
            grid_indexing.iec - 1 if grid_indexing.east_edge else grid_indexing.iec + 2
        )
        idiff = ilast - ifirst + 1

        jfirst = (
            grid_indexing.jsc + 2 if grid_indexing.south_edge else grid_indexing.jsc - 1
        )
        jlast = (
            grid_indexing.jec - 1 if grid_indexing.north_edge else grid_indexing.jec + 2
        )
        jdiff = jlast - jfirst + 1

        self._set_tmps = stencil_factory.from_dims_halo(
            func=set_tmps,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(3, 3),
        )

        self._lagrange_interpolation_y_p1 = stencil_factory.from_origin_domain(
            func=lagrange_interpolation_y_p1,
            origin=(is1, js1, 0),
            domain=(ie1 - is1 + 1, je1 - js1 + 1, grid_indexing.domain[2]),
        )

        self._lagrange_interpolation_x_p1 = stencil_factory.from_origin_domain(
            func=lagrange_interpolation_x_p1,
            origin=(is2, js2, 0),
            domain=(ie2 - is2 + 1, je2 - js2 + 1, grid_indexing.domain[2]),
        )

        origin = grid_indexing.origin_full()
        domain = grid_indexing.domain_full()
        ax_offsets = grid_indexing.axis_offsets(origin, domain)
        if npt == 0:
            d2a2c_avg_offset = -1
        else:
            d2a2c_avg_offset = 3

        self._avg_box = stencil_factory.from_dims_halo(
            func=avg_box,
            externals={"D2A2C_AVG_OFFSET": d2a2c_avg_offset},
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=(3, 3),
        )

        self._contravariant_components = stencil_factory.from_origin_domain(
            func=contravariant_components,
            origin=grid_indexing.origin_compute(add=(-1 - id_, -1 - id_, 0)),
            domain=grid_indexing.domain_compute(add=(pad, pad, 0)),
        )
        origin_edges = grid_indexing.origin_compute(add=(-3, -3, 0))
        domain_edges = grid_indexing.domain_compute(add=(6, 6, 0))
        ax_offsets_edges = grid_indexing.axis_offsets(origin_edges, domain_edges)
        self._fill_corners_x = stencil_factory.from_origin_domain(
            func=fill_corners_x,
            externals=ax_offsets_edges,
            origin=origin_edges,
            domain=domain_edges,
        )

        self._ut_main = stencil_factory.from_origin_domain(
            func=ut_main,
            origin=(ifirst, j1, 0),
            domain=(idiff, grid_indexing.domain[1] + 2, grid_indexing.domain[2]),
        )

        self._east_west_edges = stencil_factory.from_origin_domain(
            func=east_west_edges,
            externals={
                "i_end": ax_offsets_edges["i_end"],
                "i_start": ax_offsets_edges["i_start"],
                "local_je": ax_offsets_edges["local_je"],
                "local_js": ax_offsets_edges["local_js"],
            },
            origin=origin_edges,
            domain=domain_edges,
        )

        # Ydir:
        self._fill_corners_y = stencil_factory.from_origin_domain(
            func=fill_corners_y,
            externals={
                **ax_offsets_edges,
            },
            origin=origin_edges,
            domain=domain_edges,
        )

        self._north_south_edges = stencil_factory.from_origin_domain(
            func=north_south_edges,
            externals={
                "j_end": ax_offsets_edges["j_end"],
                "j_start": ax_offsets_edges["j_start"],
                "local_ie": ax_offsets_edges["local_ie"],
                "local_is": ax_offsets_edges["local_is"],
                "local_je": ax_offsets_edges["local_je"],
                "local_js": ax_offsets_edges["local_js"],
            },
            origin=origin_edges,
            domain=domain_edges,
        )

        self._vt_main = stencil_factory.from_origin_domain(
            func=vt_main,
            origin=(i1, jfirst, 0),
            domain=(grid_indexing.domain[0] + 2, jdiff, grid_indexing.domain[2]),
        )

    def __call__(self, uc, vc, u, v, ua, va, utc, vtc):
        """
        Calculate velocity vector from D-grid to A-grid to C-grid.

        Args:
            uc: C-grid x-velocity (inout)
            vc: C-grid y-velocity (inout)
            u: D-grid x-velocity (in)
            v: D-grid y-velocity (in)
            ua: A-grid x-velocity (inout)
            va: A-grid y-velocity (inout)
            utc: C-grid u * dx (inout)
            vtc: C-grid v * dy (inout)
        """
        self._set_tmps(
            self._utmp,
            self._vtmp,
            self._big_number,
        )

        self._lagrange_interpolation_y_p1(
            u,
            self._utmp,
        )

        self._lagrange_interpolation_x_p1(
            v,
            self._vtmp,
        )

        # tmp edges
        self._avg_box(
            u,
            v,
            self._utmp,
            self._vtmp,
        )

        # contra-variant components at cell center
        self._contravariant_components(
            self._utmp,
            self._vtmp,
            self._cosa_s,
            self._rsin2,
            ua,
            va,
        )
        # Fix the edges
        # Xdir:
        self._fill_corners_x(
            self._utmp,
            self._vtmp,
            ua,
            va,
        )

        self._ut_main(
            self._utmp,
            uc,
            v,
            self._cosa_u,
            self._rsin_u,
            utc,
        )

        self._east_west_edges(
            u,
            ua,
            uc,
            utc,
            self._utmp,
            v,
            self._sin_sg1,
            self._sin_sg3,
            self._cosa_u,
            self._rsin_u,
            self._dxa,
        )

        # Ydir:
        self._fill_corners_y(
            self._utmp,
            self._vtmp,
            ua,
            va,
        )

        self._north_south_edges(
            v,
            va,
            vc,
            vtc,
            self._vtmp,
            u,
            self._sin_sg2,
            self._sin_sg4,
            self._cosa_v,
            self._rsin_v,
            self._dya,
        )

        self._vt_main(
            self._vtmp,
            vc,
            u,
            self._cosa_v,
            self._rsin_v,
            vtc,
        )
