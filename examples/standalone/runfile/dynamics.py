from argparse import ArgumentParser

import mpi4py
import serialbox
import yaml

import fv3core
import fv3core._config as spec
import fv3core.stencils.fv_dynamics as fv_dynamics
import fv3core.testing
import fv3gfs.util as util
from fv3core.testing import GlobalTimer, write_to_json


if __name__ == "__main__":
    timing = GlobalTimer()
    timing.time("init")

    usage = "usage: python %(prog)s <data_dir> <timesteps> <backend>"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing data to run with",
    )
    parser.add_argument(
        "time_step",
        type=int,
        action="store",
        help="number of timesteps to execute",
    )
    parser.add_argument(
        "backend",
        type=str,
        action="store",
        help="path to the namelist",
    )
    args = parser.parse_args()
    backend = args.backend
    data_dir = args.data_dir
    time_step = args.time_step

    # # MPI stuff
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    # fv3core specific setup
    fv3core.set_backend(backend)
    fv3core.set_rebuild(False)

    # namelist setup
    spec.set_namelist(data_dir + "/input.nml")

    nml2 = yaml.safe_load(
        open(
            data_dir + "/input.yml",
            "r",
        )
    )["namelist"]
    experiment_name = yaml.safe_load(
        open(
            data_dir + "/input.yml",
            "r",
        )
    )["experiment_name"]

    # set up of helper structures
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_dir,
        "Generator_rank" + str(rank),
    )
    cube_comm = util.CubedSphereCommunicator(
        comm, util.CubedSpherePartitioner.from_namelist(nml2)
    )

    # get grid from serialized data
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    grid = fv3core.testing.TranslateGrid(grid_data, rank).python_grid()
    spec.set_grid(grid)

    # set up grid-dependent helper structures
    layout = spec.namelist.layout
    partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
    communicator = util.CubedSphereCommunicator(comm, partitioner)

    # create a state from serialized data
    savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
    driver_object = fv3core.testing.TranslateFVDynamics([grid])
    input_data = driver_object.collect_input_data(serializer, savepoint_in)
    input_data["comm"] = communicator
    state = driver_object.state_from_inputs(input_data)

    # warm-up timestep
    fv_dynamics.fv_dynamics(
        state,
        communicator,
        input_data["consv_te"],
        input_data["do_adiabatic_init"],
        input_data["bdt"],
        input_data["ptop"],
        input_data["n_split"],
        input_data["ks"],
    )
    timing.time("init")

    timing.time("mainloop")
    # Run the dynamics
    for i in range(time_step - 1):
        fv_dynamics.fv_dynamics(
            state,
            communicator,
            input_data["consv_te"],
            input_data["do_adiabatic_init"],
            input_data["bdt"],
            input_data["ptop"],
            input_data["n_split"],
            input_data["ks"],
        )

    # collect times and output simple statistics
    timing.time("mainloop")
    main_time = timing.get_totals("mainloop")["total"]
    init_time = timing.get_totals("init")["total"]
    total_time = main_time + init_time

    # write times to file
    init_times = comm.gather(init_time, root=0)
    main_times = comm.gather(main_time, root=0)
    total_times = comm.gather(total_time, root=0)
    if comm.Get_rank() == 0:
        write_to_json(
            time_step, backend, experiment_name, init_times, total_times, main_times
        )
