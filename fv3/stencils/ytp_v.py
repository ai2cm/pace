import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
from fv3._config import grid, namelist
from .yppm import compute_al,main_al,flux_intermediates, fx1_fn, final_flux, get_bl, get_br, c1, c2, c3, get_b0, is_smt5_mord5, is_smt5_most_mords
sd = utils.sd
backend = utils.backend
origin = (0, 0, 0)
halo = utils.halo




@gtscript.stencil(backend=utils.backend)
def get_flux_v_stencil(q: sd, c: sd, al: sd, rdy: sd, bl:sd, br:sd, flux: sd, mord: int):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl=bl, br=br)
        smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
        tmp = smt5[0, -1, 0] + smt5 * (smt5[0, -1, 0] == 0)
        cfl = c * rdy[0, -1, 0] if c > 0 else c * rdy
        fx0 = fx1_fn(cfl, br, b0, bl)
        flux = final_flux(c, q, fx0, tmp)


@gtscript.stencil(backend=utils.backend)
def br_bl_main(q:sd, al:sd, bl:sd, br:sd):
    with computation(PARALLEL), interval(...):
        bl = get_bl(al=al, q=q)
        br = get_br(al=al, q=q)


@gtscript.stencil(backend=utils.backend)
def br_bl_corner(br:sd, bl:sd):
    with computation(PARALLEL), interval(...):
        bl = 0
        br = 0
        bl[0, 1, 0] = 0
        br[0, 1, 0] = 0

        
def compute(c, u, v, flux):
    # This is an input argument in the Fortran code, but is never called with anything but this namelist option
    jord = namelist['hord_mt']
    if jord != 5:
        raise Exception('Currently ytp_v is only supported for hord_mt == 5')
    js3 = max(5, grid.js - 1)
    je3 = min(grid.npy - 1, grid.je+1)
    tmp_origin = (grid.is_, grid.js - 1, 0)
    bl = utils.make_storage_from_shape(u.shape, tmp_origin)
    br = utils.make_storage_from_shape(u.shape, tmp_origin)
    
    if jord < 8:
        # this not get the exact right edges
        al = compute_al(v, grid.dy, jord, grid.is_, grid.ie+1, js3, je3+1)
        br_bl_main(v, al, bl, br, origin=(grid.is_, grid.js-1, 0), domain=(grid.nic + 1, grid.njc + 2, grid.npz))
        if grid.sw_corner:
            br_bl_corner(br, bl, origin=tmp_origin, domain=grid.corner_domain())
        #    bl[grid.is_, grid.js - 1:grid.js+1, :] = 0
        #    br[grid.is_, grid.js - 1:grid.js+1, :] = 0
        if grid.se_corner:
            br_bl_corner(br, bl, origin=(grid.ie+1, grid.js - 1, 0), domain=grid.corner_domain())
        #    bl[grid.ie+1, grid.js - 1:grid.js+1, :] = 0
        #    br[grid.ie+1, grid.js - 1:grid.js+1, :] = 0
        if grid.nw_corner:
            br_bl_corner(br, bl, origin=(grid.is_, grid.je, 0), domain=grid.corner_domain())
        if grid.ne_corner:
            br_bl_corner(br, bl, origin=(grid.ie+1, grid.je, 0), domain=grid.corner_domain())
        get_flux_v_stencil(v, c, al, grid.rdy, bl, br, flux, jord, origin=(grid.is_, grid.js, 0), domain=(grid.nic + 1, grid.njc + 1, grid.npz))
       
        
