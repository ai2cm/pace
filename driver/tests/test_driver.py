import pytest
from mpi4py import MPI

from pace.util import NullTimer, Timer
from pace.driver.run import Driver
from typing import Tuple, List, Any
import fv3core._config
from fv3core.utils.null_comm import NullComm
import numpy as np

def setup_driver(dycore_only) -> Tuple[Driver, List[Any]]:

     namelist = fv3core._config.Namelist(
        layout=(1, 1),
        npx=13,
        npy=13,
        npz=79,
        ntiles=6,
        nwat=6,
        dt_atmos=225,
        a_imp=1.0,
        beta=0.0,
        consv_te=False,
        d2_bg=0.0,
        d2_bg_k1=0.2,
        d2_bg_k2=0.1,
        d4_bg=0.15,
        d_con=1.0,
        d_ext=0.0,
        dddmp=0.5,
        delt_max=0.002,
        do_sat_adj=True,
        do_vort_damp=True,
        fill=True,
        hord_dp=6,
        hord_mt=6,
        hord_tm=6,
        hord_tr=8,
        hord_vt=6,
        hydrostatic=False,
        k_split=1,
        ke_bg=0.0,
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=1,
        nord=3,
        p_fac=0.05,
        rf_fast=True,
        rf_cutoff=3000.0,
        tau=10.0,
        vtdm4=0.06,
        z_tracer=True,
        do_qa=True,
        dycore_only=dycore_only
    )
     
     comm = MPI.COMM_WORLD
     driver = Driver(
        namelist,
        comm,
        backend="numpy",
        physics_packages=["microphysics"],
        dycore_init_mode="baroclinic",
    )
     do_adiabatic_init = False
     bdt = 225.0
     args = [ 
         do_adiabatic_init,
         bdt,
     ]
     return driver, args

@pytest.mark.parametrize("sample_indices,ua_post_dycore,qv_post_dycore, qv_post_physics", [((3, 3, 6), 26.76749012814138, 3.6784598476435017e-06, 5.415568861604212e-07)])
def test_driver_runs_and_updates_data(sample_indices,ua_post_dycore, qv_post_dycore, qv_post_physics):
    
    driver, args = setup_driver(dycore_only=False)
    ti, tj, tz = sample_indices
   
    rank = driver._comm.rank
    sample_qv = driver.dycore_state.qvapor.data[ti, tj, tz]
    assert(driver.dycore_state.ua.data[ti, tj, tz] == driver.physics_state.ua.data[ti, tj, tz])
    assert(driver.physics_state.ua_t1.data[ti, tj, tz] == 0)
    assert(driver.physics_state.wmp.data[ti, tj, tz] == 0)
    driver.step_dynamics(*args)
        
    sample_qv_post_dynamics = driver.dycore_state.qvapor.data[ti, tj, tz]
    assert(sample_qv != sample_qv_post_dynamics)
    assert(driver.dycore_state.ua.data[ti, tj, tz] == driver.physics_state.ua.data[ti, tj, tz])
    if rank == 3:
        assert(driver.dycore_state.ua.data[ti, tj, tz] == ua_post_dycore)
        assert(sample_qv_post_dynamics == qv_post_dycore)
    assert(driver.physics_state.ua_t1.data[ti, tj, tz] == 0)
    assert(driver.physics_state.wmp.data[ti, tj, tz] == 0)

    driver.step_physics()
    
    assert(driver.dycore_state.qvapor.data[ti, tj, tz] != sample_qv_post_dynamics)
    assert(driver.dycore_state.ua.data[ti, tj, tz] == driver.physics_state.ua.data[ti, tj, tz])
    assert(driver.physics_state.wmp.data[ti, tj, tz] != 0)
    if rank == 3:
        assert(driver.physics_state.ua_t1.data[ti, tj, tz] ==  ua_post_dycore)
        assert(driver.dycore_state.ua.data[ti, tj, tz] == ua_post_dycore)
        assert(driver.physics_state.qvapor_t1.data[ti, tj, tz] == qv_post_dycore)
        assert(driver.dycore_state.qvapor.data[ti, tj, tz] == qv_post_physics)
        
@pytest.mark.parametrize("sample_indices,ua_post_dycore,qv_post_dycore", [((3, 3, 6), 26.76749012814138, 3.6784598476435017e-06)])
def test_driver_dycore_only(sample_indices, ua_post_dycore, qv_post_dycore):
    ti, tj, tz = sample_indices
    driver, args = setup_driver(dycore_only=True)
    with pytest.raises(AttributeError):
        driver.physics
    with pytest.raises(AttributeError):
        driver.physics_state
    with pytest.raises(AttributeError):
        driver.state_updater
    
    driver.step(*args)

    if driver._comm.rank == 3:
           assert(driver.dycore_state.ua.data[ti, tj, tz] == ua_post_dycore)
           assert(driver.dycore_state.qvapor.data[ti, tj, tz] == qv_post_dycore)
