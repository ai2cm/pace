import pytest
from mpi4py import MPI

from pace.util import NullTimer, Timer
from pace.driver.run import Driver
from typing import Tuple, List, Any
import fv3core._config
import numpy as np
from fv3core.utils.null_comm import NullComm
import contextlib


def setup_driver(dycore_only, comm) -> Tuple[Driver, List[Any]]:
     
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

# Disabled for the moment, to be run with mpirun -np 6
# running into errors in parallel. using non-MPI is all nans and data changes aren't comparable
#@pytest.mark.parametrize("sample_indices,ua_post_dycore,qv_post_dycore, qv_post_physics", [((3, 3, 6), 26.76749012814138, 3.6784598476435017e-06, 5.415568861604212e-07)])
#def test_driver_runs_and_updates_data(sample_indices,ua_post_dycore, qv_post_dycore, qv_post_physics):
#    driver, args = setup_driver(dycore_only=False, comm=MPI.COMM_WORLD)
#    ti, tj, tz = sample_indices
   
#    rank = driver._comm.rank
#    sample_qv = driver.dycore_state.qvapor.data[ti, tj, tz]
#    assert(driver.dycore_state.ua.data[ti, tj, tz] == driver.physics_state.ua.data[ti, tj, tz])
#    assert(driver.physics_state.physics_updated_ua.data[ti, tj, tz] == 0)
#    assert(driver.physics_state.wmp.data[ti, tj, tz] == 0)
#    with no_lagrangian_contributions(dynamical_core=driver.dycore):
#         driver.step_dynamics(*args)
#    print(driver.dycore_state.qvapor.data[ti, tj, tz])
#    sample_qv_post_dynamics = driver.dycore_state.qvapor.data[ti, tj, tz]
#    assert(sample_qv != sample_qv_post_dynamics)
#    assert(driver.dycore_state.ua.data[ti, tj, tz] == driver.physics_state.ua.data[ti, tj, tz])
#    if rank == 3:
#        assert(driver.dycore_state.ua.data[ti, tj, tz] == ua_post_dycore)
#        assert(sample_qv_post_dynamics == qv_post_dycore)
#    assert(driver.physics_state.physics_updated_ua.data[ti, tj, tz] == 0)
#    assert(driver.physics_state.wmp.data[ti, tj, tz] == 0)

#    driver.step_physics()
#    print(driver.dycore_state.qvapor.data[ti, tj, tz])
#    assert(driver.dycore_state.qvapor.data[ti, tj, tz] != sample_qv_post_dynamics)
#    assert(driver.dycore_state.ua.data[ti, tj, tz] == driver.physics_state.ua.data[ti, tj, tz])
#    assert(driver.physics_state.wmp.data[ti, tj, tz] != 0)
   
#    if rank == 3:
#        assert(driver.physics_state.physics_updated_ua.data[ti, tj, tz] ==  ua_post_dycore)
#        assert(driver.dycore_state.ua.data[ti, tj, tz] == ua_post_dycore)
#        assert(driver.physics_state.physics_updated_specific_humidity.data[ti, tj, tz] == qv_post_dycore)
#        assert(driver.dycore_state.qvapor.data[ti, tj, tz] == qv_post_physics)
        
def test_driver_dycore_only():
    comm =  NullComm(
         rank=0, total_ranks=6, fill_value=0.0
    )
    driver, args = setup_driver(dycore_only=True, comm=comm)
    with pytest.raises(AttributeError):
        driver.physics
    with pytest.raises(AttributeError):
        driver.physics_state
    with pytest.raises(AttributeError):
        driver.state_updater
    
   

@contextlib.contextmanager
def no_lagrangian_contributions(dynamical_core: fv3core.DynamicalCore):
    # TODO: lagrangian contributions currently cause an out of bounds iteration
    # when halo updates are disabled. Fix that bug and remove this decorator.
    # Probably requires an update to gt4py (currently v36).
    def do_nothing(*args, **kwargs):
        pass

    original_attributes = {}
    for obj in (
        dynamical_core._lagrangian_to_eulerian_obj._map_single_delz,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_pt,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_u,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_v,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_w,
        dynamical_core._lagrangian_to_eulerian_obj._map_single_delz,
    ):
        original_attributes[obj] = obj._lagrangian_contributions
        obj._lagrangian_contributions = do_nothing  # type: ignore
    for (
        obj
    ) in dynamical_core._lagrangian_to_eulerian_obj._mapn_tracer._list_of_remap_objects:
        original_attributes[obj] = obj._lagrangian_contributions
        obj._lagrangian_contributions = do_nothing  # type: ignore
    try:
        yield
    finally:
        for obj, original in original_attributes.items():
            obj._lagrangian_contributions = original

def test_driver_runs():
     comm =  NullComm(
          rank=0, total_ranks=6, fill_value=0.0
     )
     driver, args = setup_driver(dycore_only=True, comm=comm)
     with no_lagrangian_contributions(dynamical_core=driver.dycore):
          driver.step(*args)

           
