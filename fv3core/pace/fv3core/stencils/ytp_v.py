from gt4py import gtscript
from gt4py.gtscript import __INLINED, compile_assert, horizontal, region

from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core.stencils import ppm, yppm


@gtscript.function
def get_bl_br(v, dy, dya):
    """
    Args:
        v: covariant y-wind on D-grid
        dy: gridcell spacing in y-direction
        dya: gridcell spacing in y-direction on A-grid

    Returns:
        bl: ???
        br: ???
    """
    from __externals__ import i_end, i_start, j_end, j_start, jord

    if __INLINED(jord < 8):
        v_on_cell_corners = yppm.compute_al(v, dy)

        bl = v_on_cell_corners[0, 0, 0] - v[0, 0, 0]
        br = v_on_cell_corners[0, 1, 0] - v[0, 0, 0]

    else:
        dm = yppm.dm_jord8plus(v)
        v_on_cell_corners = yppm.al_jord8plus(v, dm)

        compile_assert(jord == 8)

        bl, br = yppm.blbr_jord8(v, v_on_cell_corners, dm)
        bl, br = yppm.bl_br_edges(bl, br, v, dya, v_on_cell_corners, dm)

        with horizontal(region[:, j_start + 1], region[:, j_end - 1]):
            bl, br = ppm.pert_ppm_standard_constraint_fcn(v, bl, br)

    # Zero corners
    with horizontal(
        region[i_start, j_start - 1 : j_start + 1],
        region[i_end + 1, j_start - 1 : j_start + 1],
        region[i_start, j_end : j_end + 2],
        region[i_end + 1, j_end : j_end + 2],
    ):
        bl = 0.0
        br = 0.0
    return bl, br


@gtscript.function
def advect_v_along_y(
    v: FloatField,
    vb_contra: FloatField,
    rdy: FloatFieldIJ,
    dy: FloatFieldIJ,
    dya: FloatFieldIJ,
    dt: float,
):
    """
    Advect covariant y-wind on D-grid using contravariant y-wind on cell corners.

    Named ytp_v in the original Fortran code. In the Fortran, dt is folded
    in to vb_contra and called "courant".

    Args:
        v: covariant y-wind on D-grid
        vb_contra: contravariant y-wind on cell corners
        rdy: 1 / dy
        dy: gridcell spacing in y-direction
        dya: a-grid gridcell spacing in y-direction
        dt: timestep in seconds

    Returns:
        updated_v: v having been advected by v_on_cell_corners
    """
    # Could try merging this with yppm version.

    from __externals__ import jord

    bl, br = get_bl_br(v, dy, dya)
    b0 = bl + br
    cfl = vb_contra * dt * rdy[0, -1] if vb_contra > 0 else vb_contra * dt * rdy
    fx0 = yppm.fx1_fn(cfl, br, b0, bl)

    if __INLINED(jord < 8):
        advection_mask = yppm.get_advection_mask(bl, b0, br)
    else:
        advection_mask = 1.0
    return yppm.apply_flux(vb_contra, v, fx0, advection_mask)
