from pace.util.grid import great_circle_distance_lon_lat, MetricTerms
from pace.fv3core.initialization.dycore_state import DycoreState
import pace.util as fv3util
import numpy as np
from .baroclinic import empty_numpy_dycore_state, initialize_delp, initialize_edge_pressure, local_compute_size
import pace.util.constants as constants


nhalo = fv3util.N_HALO_DEFAULT

def init_tc_state(metric_terms: MetricTerms,
    grid_config: Dict,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized to the
    FV3 tropical cyclone test case (test_case 55).

    This case involves a grid_transformation (done on metric terms)
    to locally increase resolution.
    """

    sample_quantity = metric_terms.lat
    shape = (*sample_quantity.data.shape[:2], metric_terms.ak.data.shape[0])
    nx, ny, nz = local_compute_size(shape)
    numpy_state = empty_numpy_dycore_state(shape)
    
    # Initializing to values the Fortran does for easy comparison
    numpy_state.delp[:] = 1e30
    numpy_state.delp[:nhalo, :nhalo] = 0.0
    numpy_state.delp[:nhalo, nhalo+ny:] = 0.0
    numpy_state.delp[nhalo+nx:, :nhalo] = 0.0
    numpy_state.delp[nhalo+nx:, nhalo+ny:] = 0.0
    numpy_state.pe[:] = 0.0
    #numpy_state.pt[:] = 1.0
    numpy_state.phis[:] = 1.0e30
    # numpy_state.ua[:] = 1e35
    # numpy_state.va[:] = 1e35
    # numpy_state.uc[:] = 1e30
    # numpy_state.vc[:] = 1e30
    # numpy_state.w[:] = 1.0e30
    # numpy_state.delz[:] = 1.0e25

    ps = np.zeros(numpy_state.phis.shape) # don't think we have surface pressure
    ps[:] = 101500. # don't think we have surface pressure

    #TODO why is phis = 0?
    ps, numpy_state.phis[:] = _initialize_vortex(metric_terms, numpy_state.phis)

    # TODO restart file had different ak, bk. Figure out where they came from;
    # for now, take from metric terms
    ak = metric_terms.ak.data
    bk = metric_terms.bk.data
    delp = initialize_delp(ps, ak, bk)
    ptop = 1
    numpy_state.pe[:] = initialize_edge_pressure(delp, ptop)

    # Got to line 2773 in test_help_me_figure_out_statements.

    return ""


            # do z=1,npz
            #     do j=js,je
            #         do i=is,ie
            #             delp(i,j,z) = ak(z+1)-ak(z) + ps(i,j)*(bk(z+1)-bk(z))
            #         enddo
            #     enddo
            # enddo
              
            # !Pressure
            # do j=js,je
            #     do i=is,ie
            #         pe(i,1,j) = ptop
            #     enddo
            #     do k=2,npz+1
            #         do i=is,ie
            #             pe(i,k,j) = pe(i,k-1,j) + delp(i,j,k-1)
            #         enddo
            #     enddo
            # enddo



def _initialize_vortex(metric_terms, phis):
    # this is for centering the TC
    lon_tc, lat_tc = 180., 10.
    p0 = [lon_tc * np.pi / 180., lat_tc * np.pi / 180.]
     
    dp = 1115.
    rp = 282000.
    p00 = 101500.
    
    ps = np.zeros((phis.shape))
    p_grid = metric_terms.lon_agrid.data
    for jj in range():
        for ii in range():
            p_dist = [p_grid[ii, jj, 0], p_grid[ii, jj, 1]]
            r = great_circle_distance_lon_lat(p0[0], p_dist[0], p0[1], p_dist[1], constants.RADIUS)
            ps[ii, jj] = p00 - dp * np.exp(-(r/rp)**1.5) # surface pressure?
            phis[ii, jj] = 0.
    

    return ps, phis