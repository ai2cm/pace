import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
import fv3._config as spec
import numpy as np

sd = utils.sd
origin = utils.origin

##
## Stencil Definitions
##

# Temporary flux field computations
@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def compute_f1_edge_values( flux: sd, dt2: float, velocity: sd ):
    with computation(PARALLEL), interval(...):
         flux = dt2 * velocity

@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def compute_f1_values( flux: sd, dt2: float, velocity: sd, velocity_c: sd, cosa: sd, sina: sd ):
    with computation(PARALLEL), interval(...):
         flux = dt2 * (velocity - velocity_c*cosa) / sina

# Flux field computations
@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def compute_fy_values( flux: sd, f1: sd, vort: sd ):
    with computation(PARALLEL), interval(...):
         flux = vort if f1 > 0.0 else vort[0,1,0]

@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def compute_fx_values( flux: sd, f1: sd, vort: sd ):
    with computation(PARALLEL), interval(...):
         flux = vort if f1 > 0.0 else vort[1,0,0]

# Wind speed updates
@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def update_uc( uc: sd, fy1: sd, fy: sd, rdxc: sd, ke: sd ):
    with computation(PARALLEL), interval(...):
         uc = uc + fy1*fy + rdxc*(ke[-1,0,0] - ke)

@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def update_vc( vc: sd, fx1: sd, fx: sd, rdyc: sd, ke: sd ):
    with computation(PARALLEL), interval(...):
         vc = vc - fx1*fx + rdyc*(ke[0,-1,0] - ke)


def compute(uc, vc, vort_c, ke_c, v, u, fxv, fyv, dt2):
    grid = spec.grid
    co = grid.compute_origin()

    origin = (grid.is_,grid.js,0)
    zonal_domain = (grid.npx+3,grid.npy+2,grid.npz)
    meridional_domain = (grid.npx+2,grid.npy+3,grid.npz)

    # Create storage objects for the temporary flux fields 
    fx1 = utils.make_storage_from_shape( fxv.shape, origin=origin )
    fy1 = utils.make_storage_from_shape( fyv.shape, origin=origin )

    # Compute temporary flux fields (ignoring edge values)
    compute_f1_values( fx1, dt2, u, vc, grid.cosa_v, grid.sina_v, origin=origin, domain=meridional_domain )
    compute_f1_values( fy1, dt2, v, uc, grid.cosa_u, grid.sina_u, origin=origin, domain=zonal_domain )

    # If we are not using a nested or regional grid configuration, we need to separately consider edge values
    # along the cubed sphere domain.
    if spec.namelist['grid_type'] < 3 and not grid.nested:
       edge_zonal_domain = (grid.npx+3,1,grid.npz)
       edge_meridional_domain = (1,grid.npy+3,grid.npz)

       # Compute temporary flux field edge values  
       if grid.west_edge:
          compute_f1_edge_values( fy1, dt2, v, origin=origin, domain=edge_meridional_domain )
          compute_fy_values( fyv, fy1, vort_c, origin=origin, domain=zonal_domain )
       if grid.east_edge:
          compute_f1_edge_values( fy1, dt2, v, origin=(grid.npx+2,grid.js,0), domain=edge_meridional_domain )
       if grid.south_edge:
          compute_f1_edge_values( fx1, dt2, u, origin=origin, domain=edge_zonal_domain )
       if grid.north_edge:
          compute_f1_edge_values( fx1, dt2, u, origin=(grid.is_,grid.npy+2,0), domain=edge_zonal_domain )

    # Compute main flux fields (ignoring edge values)
    compute_fx_values( fxv, fx1, vort_c, origin=origin, domain=meridional_domain )
    compute_fy_values( fyv, fy1, vort_c, origin=origin, domain=zonal_domain )

    # Update time-centered winds on C-grid 
    update_uc( uc, fy1, fyv, grid.rdxc, ke_c, origin=origin, domain=zonal_domain )
    update_vc( vc, fx1, fxv, grid.rdyc, ke_c, origin=origin, domain=meridional_domain )

