import f90nml
import numpy as np
import yaml
from mpi4py import MPI

import fv3core._config
from pace.driver.driver import Driver


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
backend = "numpy"

case_name = "/fv3core/test_data/c12_6ranks_baroclinic_dycore_microphysics"

experiment_name = yaml.safe_load(
    open(
        case_name + "/input.yml",
        "r",
    )
)["experiment_name"]
namelist = fv3core._config.Namelist.from_f90nml(f90nml.read(case_name + "/input.nml"))
output_vars = [
    "u",
    "v",
    "ua",
    "va",
    "pt",
    "delp",
    "qvapor",
    "qliquid",
    "qice",
    "qrain",
    "qsnow",
    "qgraupel",
]


driver = Driver(namelist, comm, backend, data_dir=case_name)
# TODO
do_adiabatic_init = False
# TODO derive from namelist
bdt = 225.0

for t in range(3):
    driver.step_dynamics(
        do_adiabatic_init=do_adiabatic_init,
        timestep=bdt,
    )
    driver.step_physics()
    if t % 5 == 0:
        comm.Barrier()

        output = {}

        for key in output_vars:
            getattr(driver.dycore_state, key).synchronize()
            output[key] = np.asarray(getattr(driver.dycore_state, key))
        np.save("pace_output_t_" + str(t) + "_rank_" + str(rank) + ".npy", output)
