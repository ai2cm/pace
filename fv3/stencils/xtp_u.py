import fv3.utils.gt4py_utils as utils
import fv3._config as spec
from .xppm import (
    compute_al,
    flux_intermediates,
    fx1_fn,
    final_flux,
    get_bl,
    get_br,
    is_smt5_mord5,
    is_smt5_most_mords,
    get_b0,
)
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd


@utils.stencil()
def get_flux_u_stencil_old(q: sd, c: sd, al: sd, rdx: sd, flux: sd, mord: int):
    with computation(PARALLEL), interval(...):
        bl, br, b0, tmp = flux_intermediates(q, al, mord)
        cfl = c * rdx[-1, 0, 0] if c > 0 else c * rdx
        fx0 = fx1_fn(cfl, br, b0, bl)
        # TODO: add [0, 0, 0] when gt4py bug is fixed
        flux = final_flux(c, q, fx0, tmp)  # noqa


@utils.stencil()
def get_flux_u_stencil(
    q: sd, c: sd, al: sd, rdx: sd, bl: sd, br: sd, flux: sd, mord: int
):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl=bl, br=br)
        smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
        tmp = smt5[-1, 0, 0] + smt5 * (smt5[-1, 0, 0] == 0)
        cfl = c * rdx[-1, 0, 0] if c > 0 else c * rdx
        fx0 = fx1_fn(cfl, br, b0, bl)
        # TODO: add [0, 0, 0] when gt4py bug is fixed
        flux = final_flux(c, q, fx0, tmp)  # noqa


@utils.stencil()
def br_bl_main(q: sd, al: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        # TODO: add [0, 0, 0] when gt4py bug is fixed
        bl = get_bl(al=al, q=q)  # noqa
        br = get_br(al=al, q=q)  # noqa


@utils.stencil()
def br_bl_corner(br: sd, bl: sd):
    with computation(PARALLEL), interval(...):
        bl = 0
        br = 0


def compute(c, u, v, flux):
    # This is an input argument in the Fortran code, but is never called with anything but this namelist option
    grid = spec.grid
    iord = spec.namelist["hord_mt"]
    if iord != 5:
        raise Exception("Currently ytp_v is only supported for hord_mt == 5")
    is3 = grid.is_ - 1  # max(5, grid.is_ - 1)
    ie3 = grid.ie + 1  # min(grid.npx - 1, grid.ie+1)
    tmp_origin = (is3, grid.js, 0)
    bl = utils.make_storage_from_shape(v.shape, tmp_origin)
    br = utils.make_storage_from_shape(v.shape, tmp_origin)
    corner_domain = (2, 1, grid.npz)
    if iord < 8:

        al = compute_al(u, grid.dx, iord, is3, ie3 + 1, grid.js, grid.je + 1)

        # get_flux_u_stencil_old(u, c, al, grid.rdx, flux, iord, origin=grid.compute_origin(), domain=(grid.nic + 1, grid.njc + 1, grid.npz))

        br_bl_main(
            u,
            al,
            bl,
            br,
            origin=(is3, grid.js, 0),
            domain=(ie3 - is3 + 1, grid.njc + 1, grid.npz),
        )

        if grid.sw_corner:
            br_bl_corner(
                br, bl, origin=(grid.is_ - 1, grid.js, 0), domain=corner_domain
            )
        if grid.se_corner:
            br_bl_corner(br, bl, origin=(grid.ie, grid.js, 0), domain=corner_domain)
        if grid.nw_corner:
            br_bl_corner(
                br, bl, origin=(grid.is_ - 1, grid.je + 1, 0), domain=corner_domain,
            )
        if grid.ne_corner:
            br_bl_corner(br, bl, origin=(grid.ie, grid.je + 1, 0), domain=corner_domain)

        get_flux_u_stencil(
            u,
            c,
            al,
            grid.rdx,
            bl,
            br,
            flux,
            iord,
            origin=grid.compute_origin(),
            domain=(grid.nic + 1, grid.njc + 1, grid.npz),
        )
