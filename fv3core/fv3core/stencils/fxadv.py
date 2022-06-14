from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

from fv3core.stencils.d2a2c_vect import contravariant
from pace.dsl.dace import orchestrate
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.util.grid import GridData


# TODO: the mix of local and global regions is strange here
# it's a workaround to specify DON'T do this calculation if on the tile edge
def main_uc_vc_contra(
    uc: FloatField,
    vc: FloatField,
    cosa_u: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    uc_contra: FloatField,
    vc_contra: FloatField,
):
    """
    Args:
        uc (in): covariant c-grid x-wind
        vc (in): covariant c-grid y-wind
        cosa_u (in): ???
        rsin_u (in): ???
        cosa_v (in): ???
        rsin_v (in): ???
        uc_contra (out): contravariant c-grid x-wind
        vc_contra (out): contravariant c-grid y-wind
    """
    from __externals__ import j_end, j_start, local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        utmp = uc_contra
        with horizontal(region[local_is - 1 : local_ie + 3, :]):
            # for C-grid, v must be regridded to lie at the same point as u
            v = 0.25 * (vc[-1, 0, 0] + vc + vc[-1, 1, 0] + vc[0, 1, 0])
            uc_contra = contravariant(uc, v, cosa_u, rsin_u)
        with horizontal(
            region[:, j_start - 1 : j_start + 1], region[:, j_end : j_end + 2]
        ):
            uc_contra = utmp

        with horizontal(region[:, local_js - 1 : local_je + 3]):
            # for C-grid, u must be regridded to lie at same point as v
            u = 0.25 * (uc[0, -1, 0] + uc[1, -1, 0] + uc + uc[1, 0, 0])
            vc_contra = contravariant(vc, u, cosa_v, rsin_v)


def uc_contra_y_edge(
    uc: FloatField,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    uc_contra: FloatField,
):
    """
    Args:
        uc (in):
        sin_sg1 (in):
        sin_sg3 (in):
        uc_contra (out):
    """
    from __externals__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            uc_contra = (uc / sin_sg3[-1, 0]) if (uc > 0) else (uc / sin_sg1)


def vc_contra_y_edge(
    vc: FloatField, cosa_v: FloatFieldIJ, uc_contra: FloatField, vc_contra: FloatField
):
    """
    Args:
        vc (in):
        cosa_v (in):
        uc_contra (in):
        vc_contra (inout):
    """
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
        vtmp = vc_contra
        with horizontal(
            region[i_start - 1 : i_start + 1, local_js : local_je + 2],
            region[i_end : i_end + 2, local_js : local_je + 2],
        ):
            u_contra = 0.25 * (
                uc_contra[0, -1, 0]
                + uc_contra[1, -1, 0]
                + uc_contra
                + uc_contra[1, 0, 0]
            )
            vc_contra = contravariant(vc, u_contra, cosa_v, 1.0)
        with horizontal(
            region[i_start - 1 : i_start + 1, j_start : j_start + 2],
            region[i_end : i_end + 2, j_start : j_start + 2],
            region[i_start - 1 : i_start + 1, j_end : j_end + 2],
            region[i_end : i_end + 2, j_end : j_end + 2],
        ):
            vc_contra = vtmp


def vc_contra_x_edge(
    vc: FloatField,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    vc_contra: FloatField,
):
    """
    Args:
        vc (in):
        sin_sg2 (in):
        sin_sg4 (in):
        vc_contra (out):
    """
    from __externals__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            vc_contra = (vc / sin_sg4[0, -1]) if (vc > 0) else (vc / sin_sg2)


def uc_contra_x_edge(
    uc: FloatField, cosa_u: FloatFieldIJ, vc_contra: FloatField, uc_contra: FloatField
):
    """
    Args:
        uc (in):
        cosa_u (in):
        vc_contra (in):
        uc_contra (inout):
    """
    from __externals__ import i_end, i_start, j_end, j_start, local_ie, local_is

    with computation(PARALLEL), interval(...):
        # TODO: parallel to what done for the vt_y_edge section
        utmp = uc_contra
        with horizontal(
            region[local_is : local_ie + 2, j_start - 1 : j_start + 1],
            region[local_is : local_ie + 2, j_end : j_end + 2],
        ):
            v_contra = 0.25 * (
                vc_contra[-1, 0, 0]
                + vc_contra
                + vc_contra[-1, 1, 0]
                + vc_contra[0, 1, 0]
            )
            uc_contra = contravariant(uc, v_contra, cosa_u, 1.0)
        with horizontal(
            region[i_start : i_start + 2, j_start - 1 : j_start + 1],
            region[i_start : i_start + 2, j_end : j_end + 2],
            region[i_end : i_end + 2, j_start - 1 : j_start + 1],
            region[i_end : i_end + 2, j_end : j_end + 2],
        ):
            uc_contra = utmp


def uc_contra_corners(
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    uc_contra: FloatField,
    uc_contra_copy: FloatField,
    vc_contra: FloatField,
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

    Args:
        cosa_u (in):
        cosa_v (in):
        uc (in):
        vc (in):
        uc_contra (out):
        uc_contra_copy (in): the covariant u-wind on ??? grid
        vc_contra (in):
    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 0])
        with horizontal(region[i_start + 1, j_start - 1], region[i_start + 1, j_end]):
            uc_contra = (
                uc
                - 0.25
                * cosa_u
                * (
                    vc_contra[-1, 1, 0]
                    + vc_contra[0, 1, 0]
                    + vc_contra
                    + vc[-1, 0, 0]
                    - 0.25
                    * cosa_v[-1, 0]
                    * (
                        uc_contra_copy[-1, 0, 0]
                        + uc_contra_copy[-1, -1, 0]
                        + uc_contra_copy[0, -1, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 1])
        with horizontal(region[i_start + 1, j_start], region[i_start + 1, j_end + 1]):
            damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 1])
            uc_contra = (
                uc
                - 0.25
                * cosa_u
                * (
                    vc_contra[-1, 0, 0]
                    + vc_contra
                    + vc_contra[0, 1, 0]
                    + vc[-1, 1, 0]
                    - 0.25
                    * cosa_v[-1, 1]
                    * (
                        uc_contra_copy[-1, 0, 0]
                        + uc_contra_copy[-1, 1, 0]
                        + uc_contra_copy[0, 1, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v)
        with horizontal(region[i_end, j_start - 1], region[i_end, j_end]):
            uc_contra = (
                uc
                - 0.25
                * cosa_u
                * (
                    vc_contra[0, 1, 0]
                    + vc_contra[-1, 1, 0]
                    + vc_contra[-1, 0, 0]
                    + vc
                    - 0.25
                    * cosa_v
                    * (
                        uc_contra_copy[1, 0, 0]
                        + uc_contra_copy[1, -1, 0]
                        + uc_contra_copy[0, -1, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[0, 1])
        with horizontal(region[i_end, j_start], region[i_end, j_end + 1]):
            uc_contra = (
                uc
                - 0.25
                * cosa_u
                * (
                    vc_contra
                    + vc_contra[-1, 0, 0]
                    + vc_contra[-1, 1, 0]
                    + vc[0, 1, 0]
                    - 0.25
                    * cosa_v[0, 1]
                    * (
                        uc_contra_copy[1, 0, 0]
                        + uc_contra_copy[1, 1, 0]
                        + uc_contra_copy[0, 1, 0]
                    )
                )
            ) * damp


def vc_contra_corners(
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    ut: FloatField,
    vc_contra: FloatField,
    vc_contra_copy: FloatField,
):
    """
    Args:
        cosa_u (in):
        cosa_v (in):
        uc (in):
        vc (in):
        ut (in):
        vc_contra (out):
        vc_contra_copy (in): the covariant v-wind on ??? grid
    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[0, -1] * cosa_v)
        with horizontal(region[i_start - 1, j_start + 1], region[i_end, j_start + 1]):
            vc_contra = (
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
                    * (
                        vc_contra_copy[0, -1, 0]
                        + vc_contra_copy[-1, -1, 0]
                        + vc_contra_copy[-1, 0, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[1, -1] * cosa_v)
        with horizontal(region[i_start, j_start + 1], region[i_end + 1, j_start + 1]):
            vc_contra = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut[0, -1, 0]
                    + ut
                    + ut[1, 0, 0]
                    + uc[1, -1, 0]
                    - 0.25
                    * cosa_u[1, -1]
                    * (
                        vc_contra_copy[0, -1, 0]
                        + vc_contra_copy[1, -1, 0]
                        + vc_contra_copy[1, 0, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[1, 0] * cosa_v)
        with horizontal(region[i_end + 1, j_end], region[i_start, j_end]):
            vc_contra = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut
                    + ut[0, -1, 0]
                    + ut[1, -1, 0]
                    + uc[1, 0, 0]
                    - 0.25
                    * cosa_u[1, 0]
                    * (
                        vc_contra_copy[0, 1, 0]
                        + vc_contra_copy[1, 1, 0]
                        + vc_contra_copy[1, 0, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v)
        with horizontal(region[i_end, j_end], region[i_start - 1, j_end]):
            vc_contra = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut[1, 0, 0]
                    + ut[1, -1, 0]
                    + ut[0, -1, 0]
                    + uc
                    - 0.25
                    * cosa_u
                    * (
                        vc_contra_copy[0, 1, 0]
                        + vc_contra_copy[-1, 1, 0]
                        + vc_contra_copy[-1, 0, 0]
                    )
                )
            ) * damp


"""
# Single stencil version to use when possible with gt backends
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


def fxadv_fluxes_stencil(
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    rdxa: FloatFieldIJ,
    rdya: FloatFieldIJ,
    dy: FloatFieldIJ,
    dx: FloatFieldIJ,
    crx: FloatField,
    cry: FloatField,
    x_area_flux: FloatField,
    y_area_flux: FloatField,
    uc_contra: FloatField,
    vc_contra: FloatField,
    dt: float,
):
    """
    Args:
        sin_sg1 (in):
        sin_sg2 (in):
        sin_sg3 (in):
        sin_sg4 (in):
        rdxa (in):
        rdya (in):
        dy (in):
        dx (in):
        crx (out):
        cry (out):
        x_area_flux (out):
        y_area_flux (out):
        uc_contra (in):
        vc_contra (in):
    """
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        prod = dt * uc_contra
        with horizontal(region[local_is : local_ie + 2, :]):
            if prod > 0:
                crx = prod * rdxa[-1, 0]
                x_area_flux = dy * prod * sin_sg3[-1, 0]
            else:
                crx = prod * rdxa
                x_area_flux = dy * prod * sin_sg1
        prod = dt * vc_contra
        with horizontal(region[:, local_js : local_je + 2]):
            if prod > 0:
                cry = prod * rdya[0, -1]
                y_area_flux = dx * prod * sin_sg4[0, -1]
            else:
                cry = prod * rdya
                y_area_flux = dx * prod * sin_sg2


class FiniteVolumeFluxPrep:
    """
    A large section of code near the beginning of Fortran's d_sw subroutinw
    Known in this repo as FxAdv,
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )
        grid_indexing = stencil_factory.grid_indexing
        self._tile_interior = not (
            grid_indexing.west_edge
            or grid_indexing.east_edge
            or grid_indexing.north_edge
            or grid_indexing.south_edge
        )
        self._dx = grid_data.dx
        self._dy = grid_data.dy
        self._rdxa = grid_data.rdxa
        self._rdya = grid_data.rdya
        self._cosa_u = grid_data.cosa_u
        self._cosa_v = grid_data.cosa_v
        self._rsin_u = grid_data.rsin_u
        self._rsin_v = grid_data.rsin_v
        self._sin_sg1 = grid_data.sin_sg1
        self._sin_sg2 = grid_data.sin_sg2
        self._sin_sg3 = grid_data.sin_sg3
        self._sin_sg4 = grid_data.sin_sg4
        origin = grid_indexing.origin_full()
        domain = grid_indexing.domain_full()
        ax_offsets = grid_indexing.axis_offsets(origin, domain)
        kwargs = {"externals": ax_offsets, "origin": origin, "domain": domain}
        origin_corners = grid_indexing.origin_full(add=(1, 1, 0))
        domain_corners = grid_indexing.domain_full(add=(-1, -1, 0))
        corner_offsets = grid_indexing.axis_offsets(origin_corners, domain_corners)
        kwargs_corners = {
            "externals": corner_offsets,
            "origin": origin_corners,
            "domain": domain_corners,
        }
        self._main_uc_vc_contra_stencil = stencil_factory.from_origin_domain(
            main_uc_vc_contra, **kwargs
        )
        self._uc_contra_y_edge_stencil = stencil_factory.from_origin_domain(
            uc_contra_y_edge, **kwargs
        )
        self._vc_contra_y_edge_stencil = stencil_factory.from_origin_domain(
            vc_contra_y_edge, **kwargs
        )
        self._vc_contra_x_edge_stencil = stencil_factory.from_origin_domain(
            vc_contra_x_edge, **kwargs
        )
        self._uc_contra_x_edge_stencil = stencil_factory.from_origin_domain(
            uc_contra_x_edge, **kwargs
        )
        self._uc_contra_corners_stencil = stencil_factory.from_origin_domain(
            uc_contra_corners, **kwargs_corners
        )
        self._vc_contra_corners_stencil = stencil_factory.from_origin_domain(
            vc_contra_corners, **kwargs_corners
        )
        self._fxadv_fluxes_stencil = stencil_factory.from_origin_domain(
            fxadv_fluxes_stencil, **kwargs
        )
        # self._set_nans = get_set_nan_func(
        #     grid_indexing,
        #     dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
        #     n_halo=((2, 2), (2, 2)),
        # )

    def __call__(
        self,
        uc,
        vc,
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        uc_contra,
        vc_contra,
        dt,
    ):
        """
        Updates flux operators and courant numbers for fvtp2d
        To start off D_SW after the C-grid winds have been advanced half a timestep,
        and and compute finite volume transport on the D-grid (e.g.Putman and Lin 2007),
        this module prepares terms such as parts of equations 7 and 13 in Putnam and
        Lin, 2007, that get consumed by fvtp2d and ppm methods.

        Args:
            uc (in): covariant x-velocity on the C-grid
            vc (in): covariant y-velocity on the C-grid
            crx (out): Courant number, x direction
            cry (out): Courant number, y direction
            x_area_flux (out): flux of area in x-direction, in units of m^2
            y_area_flux (out): flux of area in y-direction, in units of m^2
            uc_contra (out): contravariant x-velocity on C-grid
            vc_contra (out): contravariant y-velocity on C-grid
            dt (in): acoustic timestep in seconds
        """

        # in the original Fortran code, uc_contra is named ut and vc_contra is vt

        self._main_uc_vc_contra_stencil(
            uc,
            vc,
            self._cosa_u,
            self._rsin_u,
            self._cosa_v,
            self._rsin_v,
            uc_contra,
            vc_contra,
        )
        if not self._tile_interior:
            self._uc_contra_y_edge_stencil(uc, self._sin_sg1, self._sin_sg3, uc_contra)
            self._vc_contra_y_edge_stencil(
                vc,
                self._cosa_v,
                uc_contra,
                vc_contra,
            )
            self._vc_contra_x_edge_stencil(vc, self._sin_sg2, self._sin_sg4, vc_contra)
            self._uc_contra_x_edge_stencil(
                uc,
                self._cosa_u,
                vc_contra,
                uc_contra,
            )
            # NOTE: this is aliasing memory
            self._uc_contra_corners_stencil(
                self._cosa_u,
                self._cosa_v,
                uc,
                vc,
                uc_contra,
                uc_contra,
                vc_contra,
            )
            # NOTE: this is aliasing memory
            self._vc_contra_corners_stencil(
                self._cosa_u,
                self._cosa_v,
                uc,
                vc,
                uc_contra,
                vc_contra,
                vc_contra,
            )
        self._fxadv_fluxes_stencil(
            self._sin_sg1,
            self._sin_sg2,
            self._sin_sg3,
            self._sin_sg4,
            self._rdxa,
            self._rdya,
            self._dy,
            self._dx,
            crx,
            cry,
            x_area_flux,
            y_area_flux,
            uc_contra,
            vc_contra,
            dt,
        )
        # self._set_nans(uc_contra)
        # self._set_nans(vc_contra)


# -------------------- DEPRECATED CORNERS-----------------
# TODO: Remove this when satisfied with this file, below is
# another implementation option:
# Using 1 function with different sets of externals
# Now that we are using a class here, this could work and
# be performant if all external variations are initialized
# as different stencil objects.
# Or if gt4py adds feature to assign index offsets with runtime integers,
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
