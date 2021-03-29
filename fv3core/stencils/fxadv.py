import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField, FloatFieldIJ


# TODO: the mix of local and global regions is strange here
# it's a workaround to specify DON'T do this calculation if on the tile edge
# check that the fortran is correct
@gtstencil
def main_ut(
    uc: FloatField,
    vc: FloatField,
    cosa_u: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    ut: FloatField,
):
    from __externals__ import j_end, j_start, local_ie, local_is

    with computation(PARALLEL), interval(...):
        utmp = ut
        with horizontal(region[local_is - 1 : local_ie + 3, :]):
            ut = (
                uc - 0.25 * cosa_u * (vc[-1, 0, 0] + vc + vc[-1, 1, 0] + vc[0, 1, 0])
            ) * rsin_u
        with horizontal(
            region[:, j_start - 1 : j_start + 1], region[:, j_end : j_end + 2]
        ):
            ut = utmp


# TODO: the mix of local and global regions is strange here
# it's a workaround to specify DON'T do this calculation if on the tile edge
# check that the fortran is correct
@gtstencil
def main_vt(
    uc: FloatField,
    vc: FloatField,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    vt: FloatField,
):
    from __externals__ import j_end, j_start, local_je, local_js

    with computation(PARALLEL), interval(...):
        vtmp = vt
        with horizontal(region[:, local_js - 1 : local_je + 3]):
            vt = (
                vc - 0.25 * cosa_v * (uc[0, -1, 0] + uc[1, -1, 0] + uc + uc[1, 0, 0])
            ) * rsin_v
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            vt = vtmp


@gtstencil
def ut_y_edge(
    uc: FloatField,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    ut: FloatField,
    dt: float,
):
    from __externals__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            ut = (uc / sin_sg3[-1, 0]) if (uc * dt > 0) else (uc / sin_sg1)


@gtstencil
def ut_x_edge(uc: FloatField, cosa_u: FloatFieldIJ, vt: FloatField, ut: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start, local_ie, local_is

    with computation(PARALLEL), interval(...):
        # TODO: parallel to what done for the vt_y_edge section
        utmp = ut
        with horizontal(
            region[local_is : local_ie + 2, j_start - 1 : j_start + 1],
            region[local_is : local_ie + 2, j_end : j_end + 2],
        ):
            ut = uc - 0.25 * cosa_u * (vt[-1, 0, 0] + vt + vt[-1, 1, 0] + vt[0, 1, 0])
        with horizontal(
            region[i_start : i_start + 2, j_start - 1 : j_start + 1],
            region[i_start : i_start + 2, j_end : j_end + 2],
            region[i_end : i_end + 2, j_start - 1 : j_start + 1],
            region[i_end : i_end + 2, j_end : j_end + 2],
        ):
            ut = utmp


@gtstencil
def vt_y_edge(vc: FloatField, cosa_v: FloatFieldIJ, ut: FloatField, vt: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start, local_je, local_js

    # This works for 6 ranks, but not 54:
    # with horizontal(region[i_start - 1: i_start + 1, j_start + 2:j_end], \
    #                region[i_end : i_end + 2, j_start+2:j_end]):
    #    vt = vc - 0.25 * cosa_v * (
    #        ut[0, -1, 0] + ut[1, -1, 0] + ut + ut[1, 0, 0]
    #    )
    # original bounds with stencil calls
    # j1 = grid().js + 2 if grid().south_edge else grid().js
    # j2 = grid().je if grid().north_edge else grid().je + 2
    # TODO: this is a hack, copying vt to vtmp to 'correct' the edges
    # Can we *just* apply edge calculations in the correct regions without overcomputing
    # rank 0, 1, 2: local_js + 2:local_je + 2
    # rank 3, 4, 5: local_js:local_je + 2
    # rank 6, 7, 8: local_js:local_je
    with computation(PARALLEL), interval(...):
        vtmp = vt
        with horizontal(
            region[i_start - 1 : i_start + 1, local_js : local_je + 2],
            region[i_end : i_end + 2, local_js : local_je + 2],
        ):
            vt = vc - 0.25 * cosa_v * (ut[0, -1, 0] + ut[1, -1, 0] + ut + ut[1, 0, 0])
        with horizontal(
            region[i_start - 1 : i_start + 1, j_start : j_start + 2],
            region[i_end : i_end + 2, j_start : j_start + 2],
            region[i_start - 1 : i_start + 1, j_end : j_end + 2],
            region[i_end : i_end + 2, j_end : j_end + 2],
        ):
            vt = vtmp


@gtstencil
def vt_x_edge(
    vc: FloatField,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    vt: FloatField,
    dt: float,
):
    from __externals__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            vt = (vc / sin_sg4[0, -1]) if (vc * dt > 0) else (vc / sin_sg2)


@gtstencil
def ut_corners(
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    ut: FloatField,
    vt: FloatField,
):

    """
    The following code (and vt_corners) solves a 2x2 system to
    get the interior parallel-to-edge uc,vc values near the corners
    (ex: for the sw corner ut(2,1) and vt(1,2) are solved for simultaneously).
    It then computes the halo uc, vc values so as to be consistent with the
    computations on the facing panel.
    The system solved is:
       ut(2,1) = uc(2,1) - avg(vt)*cosa_u(2,1)
       vt(1,2) = vc(1,2) - avg(ut)*cosa_v(1,2)
       in which avg(vt) includes vt(1,2) and avg(ut) includes ut(2,1)

    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 0])
        with horizontal(region[i_start + 1, j_start - 1], region[i_start + 1, j_end]):
            ut = (
                uc
                - 0.25
                * cosa_u
                * (
                    vt[-1, 1, 0]
                    + vt[0, 1, 0]
                    + vt
                    + vc[-1, 0, 0]
                    - 0.25
                    * cosa_v[-1, 0]
                    * (ut[-1, 0, 0] + ut[-1, -1, 0] + ut[0, -1, 0])
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 1])
        with horizontal(region[i_start + 1, j_start], region[i_start + 1, j_end + 1]):
            damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 1])
            ut = (
                uc
                - 0.25
                * cosa_u
                * (
                    vt[-1, 0, 0]
                    + vt
                    + vt[0, 1, 0]
                    + vc[-1, 1, 0]
                    - 0.25 * cosa_v[-1, 1] * (ut[-1, 0, 0] + ut[-1, 1, 0] + ut[0, 1, 0])
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v)
        with horizontal(region[i_end, j_start - 1], region[i_end, j_end]):
            ut = (
                uc
                - 0.25
                * cosa_u
                * (
                    vt[0, 1, 0]
                    + vt[-1, 1, 0]
                    + vt[-1, 0, 0]
                    + vc
                    - 0.25 * cosa_v * (ut[1, 0, 0] + ut[1, -1, 0] + ut[0, -1, 0])
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[0, 1])
        with horizontal(region[i_end, j_start], region[i_end, j_end + 1]):
            ut = (
                uc
                - 0.25
                * cosa_u
                * (
                    vt
                    + vt[-1, 0, 0]
                    + vt[-1, 1, 0]
                    + vc[0, 1, 0]
                    - 0.25 * cosa_v[0, 1] * (ut[1, 0, 0] + ut[1, 1, 0] + ut[0, 1, 0])
                )
            ) * damp


@gtstencil
def vt_corners(
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    ut: FloatField,
    vt: FloatField,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[0, -1] * cosa_v)
        with horizontal(region[i_start - 1, j_start + 1], region[i_end, j_start + 1]):
            vt = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut[1, -1, 0]
                    + ut[1, 0, 0]
                    + ut
                    + uc[0, -1, 0]
                    - 0.25
                    * cosa_u[0, -1]
                    * (vt[0, -1, 0] + vt[-1, -1, 0] + vt[-1, 0, 0])
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[1, -1] * cosa_v)
        with horizontal(region[i_start, j_start + 1], region[i_end + 1, j_start + 1]):
            vt = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut[0, -1, 0]
                    + ut
                    + ut[1, 0, 0]
                    + uc[1, -1, 0]
                    - 0.25 * cosa_u[1, -1] * (vt[0, -1, 0] + vt[1, -1, 0] + vt[1, 0, 0])
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[1, 0] * cosa_v)
        with horizontal(region[i_end + 1, j_end], region[i_start, j_end]):
            vt = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut
                    + ut[0, -1, 0]
                    + ut[1, -1, 0]
                    + uc[1, 0, 0]
                    - 0.25 * cosa_u[1, 0] * (vt[0, 1, 0] + vt[1, 1, 0] + vt[1, 0, 0])
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v)
        with horizontal(region[i_end, j_end], region[i_start - 1, j_end]):
            vt = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut[1, 0, 0]
                    + ut[1, -1, 0]
                    + ut[0, -1, 0]
                    + uc
                    - 0.25 * cosa_u * (vt[0, 1, 0] + vt[-1, 1, 0] + vt[-1, 0, 0])
                )
            ) * damp


@gtscript.function
def ra_x_func(area, xfx_adv):
    return area + xfx_adv - xfx_adv[1, 0, 0]


@gtscript.function
def ra_y_func(area, yfx_adv):
    return area + yfx_adv - yfx_adv[0, 1, 0]


"""
@gtstencil()
def fxadv_stencil(
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    ut: FloatField,
    vt: FloatField,
    dt: float,
):
    Updates flux operators and courant numbers for fvtp2d

    To kick off D_SW after the C-grid winds have been advanced half a timestep,
    and and compute finite volume transport on the D-grid (e.g.Putman and Lin 2007),
    this module prepares terms such as parts of equations 7 and 13 in Putnam and Lin,
    2007, that get consumed by fvtp2d and ppm methods.

    Args:
        uc: x-velocity on the C-grid (in)
        vc: y-velocity on the C-grid (in)
        crx_adv: Courant number, x direction(inout)
        cry_adv: Courant number, y direction(inout)
        xfx_adv: Finite volume flux form operator in x direction (inout)
        yfx_adv: Finite volume flux form operator in y direction (inout)
        ut: temporary x-velocity transformed from C-grid to D-grid equivalent(?) (inout)
        vt: temporary y-velocity transformed from C-grid to D-grid equivalent(?) (inout)
        dt: timestep in seconds
    Grid variable inputs:
        cosa_u, cosa_v, rsin_u, rsin_v, sin_sg1,sin_sg2, sin_sg3, sin_sg4

    with computation(PARALLEL), interval(...):
        ut = main_ut(uc, vc, cosa_u, rsin_u, ut)
        ut = ut_y_edge(uc, sin_sg1, sin_sg3, ut, dt)
        vt = main_vt(uc, vc, cosa_v, rsin_v, vt)
        vt = vt_y_edge(vc, cosa_v, ut, vt)
        vt = vt_x_edge(vc, sin_sg2, sin_sg4, vt, dt)
        ut = ut_x_edge(uc, cosa_u, vt, ut)
        ut = ut_corners(uc, vc, cosa_u, cosa_v, ut, vt)
        vt = vt_corners(uc, vc, cosa_u, cosa_v, ut, vt)
"""


@gtstencil()
def fxadv_fluxes_stencil(
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    rdxa: FloatFieldIJ,
    rdya: FloatFieldIJ,
    dy: FloatFieldIJ,
    dx: FloatFieldIJ,
    crx_adv: FloatField,
    cry_adv: FloatField,
    xfx_adv: FloatField,
    yfx_adv: FloatField,
    ut: FloatField,
    vt: FloatField,
    dt: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        prod = dt * ut
        with horizontal(region[local_is : local_ie + 2, :]):
            crx_adv = prod * rdxa[-1, 0] if prod > 0 else prod * rdxa
            xfx_adv = dy * prod * sin_sg3[-1, 0] if prod > 0 else dy * prod * sin_sg1
        prod = dt * vt
        with horizontal(region[:, local_js : local_je + 2]):
            cry_adv = prod * rdya[0, -1] if prod > 0 else prod * rdya
            yfx_adv = dx * prod * sin_sg4[0, -1] if prod > 0 else dx * prod * sin_sg2


@gtstencil()
def flux_divergence_area(
    area: FloatFieldIJ,
    xfx_adv: FloatField,
    yfx_adv: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
):
    """Compute the area with flux divergence applied
     Args:
       xfx_adv: Finite volume flux form operator in x direction (in)
       yfx_adv: Finite volume flux form operator in y direction (in)
       ra_x: Area increased in the x direction due to flux divergence (inout)
       ra_y: Area increased in the y direction due to flux divergence (inout)
    Grid variable inputs:
       area
    """
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        with horizontal(region[local_is : local_ie + 2, :]):
            ra_x = ra_x_func(area, xfx_adv)
        with horizontal(region[:, local_js : local_je + 2]):
            ra_y = ra_y_func(area, yfx_adv)


def compute(uc, vc, crx_adv, cry_adv, xfx_adv, yfx_adv, ut, vt, ra_x, ra_y, dt):
    grid = spec.grid
    main_ut(
        uc,
        vc,
        grid.cosa_u,
        grid.rsin_u,
        ut,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )
    ut_y_edge(
        uc,
        grid.sin_sg1,
        grid.sin_sg3,
        ut,
        dt,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )
    main_vt(
        uc,
        vc,
        grid.cosa_v,
        grid.rsin_v,
        vt,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )
    vt_y_edge(
        vc,
        grid.cosa_v,
        ut,
        vt,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )
    vt_x_edge(
        vc,
        grid.sin_sg2,
        grid.sin_sg4,
        vt,
        dt,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )
    ut_x_edge(
        uc,
        grid.cosa_u,
        vt,
        ut,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )
    ut_corners(
        grid.cosa_u,
        grid.cosa_v,
        uc,
        vc,
        ut,
        vt,
        origin=(1, 1, 0),
        domain=grid.domain_shape_full(add=(-1, -1, 0)),
    )
    vt_corners(
        grid.cosa_u,
        grid.cosa_v,
        uc,
        vc,
        ut,
        vt,
        origin=(1, 1, 0),
        domain=grid.domain_shape_full(add=(-1, -1, 0)),
    )
    fxadv_fluxes_stencil(
        grid.sin_sg1,
        grid.sin_sg2,
        grid.sin_sg3,
        grid.sin_sg4,
        grid.rdxa,
        grid.rdya,
        grid.dy,
        grid.dx,
        crx_adv,
        cry_adv,
        xfx_adv,
        yfx_adv,
        ut,
        vt,
        dt,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )
    flux_divergence_area(
        grid.area,
        xfx_adv,
        yfx_adv,
        ra_x,
        ra_y,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )


# -------------------- DEPRECATED CORNERS-----------------
# Using 1 function with different sets of externals
# Unlikely to use as different externals in single stencil version
# but if gt4py adds feature to assign index offsets with runtime integers,
# this might be useful.
# Note, it changes the order of operatons slightly and yields 1e-15 errors
# @gtscript.function
# def corner_ut_function(uc: FloatField, vc: FloatField, ut: FloatField,
#              vt: FloatField, cosa_u: FloatField, cosa_v: FloatField):
#     from __externals__ import ux, uy, vi, vj, vx, vy
#     with computation(PARALLEL), interval(...):
#         ut = (
#             (
#                 uc
#                 - 0.25
#                 * cosa_u
#                 * (
#                      vt[vi, vy, 0]
#                     + vt[vx, vy, 0]
#                     + vt[vx, vj, 0]
#                     + vc[vi, vj, 0]
#                     - 0.25
#                     * cosa_v[vi, vj, 0]
#                     * (ut[ux, 0, 0] + ut[ux, uy, 0] + ut[0, uy, 0])
#                 )
#             )
#             * 1.0
#             / (1.0 - 0.0625 * cosa_u * cosa_v[vi, vj, 0])
#         )
#
#
# def corner_ut_stencil(uc: FloatField, vc: FloatField, ut: FloatField, \
#     vt: FloatField, cosa_u: FloatField, cosa_v: FloatField):
#     from __externals__ import ux, uy, vi, vj, vx, vy
#
#     with computation(PARALLEL), interval(...):
#         ut = (
#             (
#                 uc
#                 - 0.25
#                 * cosa_u
#                 * (
#                     vt[vi, vy, 0]
#                     + vt[vx, vy, 0]
#                     + vt[vx, vj, 0]
#                     + vc[vi, vj, 0]
#                     - 0.25
#                     * cosa_v[vi, vj, 0]
#                     * (ut[ux, 0, 0] + ut[ux, uy, 0] + ut[0, uy, 0])
#                 )
#             )
#             * 1.0
#             / (1.0 - 0.0625 * cosa_u * cosa_v[vi, vj, 0])
#         )
#
#
# # for the non-stencil version of filling corners
# def get_damp(cosa_u, cosa_v, ui, uj, vi, vj):
#     return 1.0 / (1.0 - 0.0625 * cosa_u[ui, uj, :] * cosa_v[vi, vj, :])
#
#
# def index_offset(lower, u, south=True):
#     if lower == u:
#         offset = 1
#     else:
#         offset = -1
#     if south:
#         offset *= -1
#     return offset
#
#
# def corner_ut(
#     uc,
#     vc,
#     ut,
#     vt,
#     cosa_u,
#     cosa_v,
#     ui,
#     uj,
#     vi,
#     vj,
#     west,
#     lower,
#     south=True,
#     vswitch=False,
# ):
#     if vswitch:
#         lowerfactor = 1 if lower else -1
#     else:
#         lowerfactor = 1
#     vx = vi + index_offset(west, False, south) * lowerfactor
#     ux = ui + index_offset(west, True, south) * lowerfactor
#     vy = vj + index_offset(lower, False, south) * lowerfactor
#     uy = uj + index_offset(lower, True, south) * lowerfactor
#     if stencil_corner:
#         decorator = gtscript.stencil(
#             backend=global_config.get_backend(),
#             externals={
#                 "vi": vi - ui,
#                 "vj": vj - uj,
#                 "ux": ux - ui,
#                 "uy": uy - uj,
#                 "vx": vx - ui,
#                 "vy": vy - uj,
#             },
#             rebuild=global_config.get_rebuild(),
#         )
#         corner_stencil = decorator(corner_ut_stencil)
#         corner_stencil(
#             uc,
#             vc,
#             ut,
#             vt,
#             cosa_u,
#             cosa_v,
#             origin=(ui, uj, 0),
#             domain=(1, 1, grid.npz),
#         )
#     else:
#         damp = get_damp(cosa_u, cosa_v, ui, uj, vi, vj)
#         ut[ui, uj, :] = (
#             uc[ui, uj, :]
#             - 0.25
#             * cosa_u[ui, uj, :]
#             * (
#                 vt[vi, vy, :]
#                 + vt[vx, vy, :]
#                 + vt[vx, vj, :]
#                 + vc[vi, vj, :]
#                 - 0.25
#                 * cosa_v[vi, vj, :]
#                 * (ut[ux, uj, :] + ut[ux, uy, :] + ut[ui, uy, :])
#             )
#         ) * damp
#
#
# def sw_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
#     t = grid.is_ + 1
#     n = grid.is_
#     z = grid.is_ - 1
#     corner_ut(uc, vc, ut, vt, cosa_u, cosa_v, t, z, n, z, west=True, lower=True)
#     corner_ut(
#       vc, uc, vt, ut, cosa_v, cosa_u, z, t, z, n, west=True, lower=True, vswitch=True
#     )
#     corner_ut(uc, vc, ut, vt, cosa_u, cosa_v, t, n, n, t, west=True, lower=False)
#     corner_ut(
#       vc, uc, vt, ut, cosa_v, cosa_u, n, t, t, n, west=True, lower=False, vswitch=True
#     )
#
#
# def se_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
#     t = grid.js + 1
#     n = grid.js
#     z = grid.js - 1
#     corner_ut(
#         uc,
#         vc,
#         ut,
#         vt,
#         cosa_u,
#         cosa_v,
#         grid.ie,
#         z,
#         grid.ie,
#         z,
#         west=False,
#         lower=True,
#     )
#     corner_ut(
#         vc,
#         uc,
#         vt,
#         ut,
#         cosa_v,
#         cosa_u,
#         grid.ie + 1,
#         t,
#         grid.ie + 2,
#         n,
#         west=False,
#         lower=True,
#         vswitch=True,
#     )
#     corner_ut(
#         uc,
#         vc,
#         ut,
#         vt,
#         cosa_u,
#         cosa_v,
#         grid.ie,
#         n,
#         grid.ie,
#         t,
#         west=False,
#         lower=False,
#     )
#     corner_ut(
#         vc,
#         uc,
#         vt,
#         ut,
#         cosa_v,
#         cosa_u,
#         grid.ie,
#         t,
#         grid.ie,
#         n,
#         west=False,
#         lower=False,
#         vswitch=True,
#     )
#
#
# def ne_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
#     corner_ut(
#         uc,
#         vc,
#         ut,
#         vt,
#         cosa_u,
#         cosa_v,
#         grid.ie,
#         grid.je + 1,
#         grid.ie,
#         grid.je + 2,
#         west=False,
#         lower=False,
#     )
#     corner_ut(
#         vc,
#         uc,
#         vt,
#         ut,
#         cosa_v,
#         cosa_u,
#         grid.ie + 1,
#         grid.je,
#         grid.ie + 2,
#         grid.je,
#         west=False,
#         lower=False,
#         south=False,
#         vswitch=True,
#     )
#     corner_ut(
#         uc,
#         vc,
#         ut,
#         vt,
#         cosa_u,
#         cosa_v,
#         grid.ie,
#         grid.je,
#         grid.ie,
#         grid.je,
#         west=False,
#         lower=True,
#     )
#     corner_ut(
#         vc,
#         uc,
#         vt,
#         ut,
#         cosa_v,
#         cosa_u,
#         grid.ie,
#         grid.je,
#         grid.ie,
#         grid.je,
#         west=False,
#         lower=True,
#         south=False,
#         vswitch=True,
#     )
#
#
# def nw_corner(uc, vc, ut, vt, cosa_u, cosa_v, corner_shape):
#     t = grid.js + 1
#     n = grid.js
#     z = grid.js - 1
#     corner_ut(
#         uc,
#         vc,
#         ut,
#         vt,
#         cosa_u,
#         cosa_v,
#         t,
#         grid.je + 1,
#         n,
#         grid.je + 2,
#         west=True,
#         lower=False,
#     )
#     corner_ut(
#         vc,
#         uc,
#         vt,
#         ut,
#         cosa_v,
#         cosa_u,
#         z,
#         grid.je,
#         z,
#         grid.je,
#         west=True,
#         lower=False,
#         south=False,
#         vswitch=True,
#     )
#     corner_ut(
#         uc,
#         vc,
#         ut,
#         vt,
#         cosa_u,
#         cosa_v,
#         t,
#         grid.je,
#         n,
#         grid.je,
#         west=True,
#         lower=True,
#     )
#     corner_ut(
#         vc,
#         uc,
#         vt,
#         ut,
#         cosa_v,
#         cosa_u,
#         n,
#         grid.je,
#         t,
#         grid.je,
#         west=True,
#         lower=True,
#         south=False,
#         vswitch=True,
#     )
