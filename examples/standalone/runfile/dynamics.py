import json
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import serialbox
import yaml
from mpi4py import MPI

import fv3core
import fv3core._config as spec
import fv3core.stencils.fv_dynamics as fv_dynamics
import fv3core.testing
import fv3gfs.util as util


def parse_args():
    usage = "usage: python %(prog)s <data_dir> <timesteps> <backend> <hash>"
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
    parser.add_argument(
        "hash",
        type=str,
        action="store",
        help="git hash to store",
    )

    return parser.parse_args()


def set_experiment_info(experiment_name, time_step, backend, git_hash):
    experiment = {}
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    experiment["setup"] = {}
    experiment["setup"]["timestamp"] = dt_string
    experiment["setup"]["dataset"] = experiment_name
    experiment["setup"]["timesteps"] = time_step
    experiment["setup"]["hash"] = git_hash
    experiment["setup"]["version"] = "python/" + backend
    experiment["times"] = {}
    return experiment


def gather_timing_statistics(timer, experiment, comm, root=0):
    is_root = comm.Get_rank() == root
    recvbuf = np.array(0.0)
    for name, value in timer.times.items():
        if is_root:
            print(name)
            experiment["times"][name] = {}
            experiment["times"][name]["hits"] = int(timer.hits[name])
        for label, op in [
            ("minimum", MPI.MIN),
            ("maximum", MPI.MAX),
            ("mean", MPI.SUM),
        ]:
            comm.Reduce(np.array(value), recvbuf, op=op)
            if is_root:
                if label == "mean":
                    recvbuf /= comm.Get_size()
                print(f"    {label}: {recvbuf}")
                experiment["times"][name][label] = float(recvbuf)


def write_global_timings(experiment, filename, comm, root=0):
    is_root = comm.Get_rank() == root
    if is_root:
        with open(filename + ".json", "w") as outfile:
            json.dump(experiment, outfile, sort_keys=True, indent=4)


if __name__ == "__main__":
    timer = util.Timer()
    timer.start("total")
    with timer.clock("initialization"):
        args = parse_args()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        fv3core.set_backend(args.backend)
        fv3core.set_rebuild(False)

        spec.set_namelist(args.data_dir + "/input.nml")

        experiment_name = yaml.safe_load(
            open(
                args.data_dir + "/input.yml",
                "r",
            )
        )["experiment_name"]

        # set up of helper structures
        serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read,
            args.data_dir,
            "Generator_rank" + str(rank),
        )
        cube_comm = util.CubedSphereCommunicator(
            comm,
            util.CubedSpherePartitioner(util.TilePartitioner(spec.namelist.layout)),
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

        # warm-up timestep.
        # We're intentionally not passing the timer here to exclude
        # warmup/compilation from the internal timers
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

    with timer.clock("mainloop"):
        for i in range(args.time_step - 1):
            fv_dynamics.fv_dynamics(
                state,
                communicator,
                input_data["consv_te"],
                input_data["do_adiabatic_init"],
                input_data["bdt"],
                input_data["ptop"],
                input_data["n_split"],
                input_data["ks"],
                timer,
            )

    timer.stop("total")
    # collect times and output simple statistics
    print("Gathering Times")
    experiment = set_experiment_info(
        experiment_name, args.time_step, args.backend, args.hash
    )
    gather_timing_statistics(timer, experiment, comm)
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d-%H-%M-%S")
    write_global_timings(experiment, filename, comm)
