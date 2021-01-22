import json
import pathlib
from datetime import datetime
from statistics import mean, median
from typing import List

import git
import mpi4py


class GlobalTimer(object):
    """
    Class to accumulate timings for named operations
    """

    def __init__(self):
        self.is_on = {}
        self.times = {}
        self.disabled = {}

    def toggle(self, name: str):
        """Enable or disable the timer of a named globally."""
        if name not in self.disabled:
            self.disabled[name] = False
        self.disabled[name] = not self.disabled[name]

    def time(self, name: str):
        """Start or stop a given timer of a named operation."""
        if name not in self.disabled or not self.disabled[name]:
            if name not in self.is_on:
                self.is_on[name] = False
                self.times[name] = {}
                self.times[name]["counter"] = 0
            if self.is_on[name]:
                current_time = mpi4py.MPI.Wtime()
                elapsed = current_time - self.times[name]["time"]
                self.times[name]["total"] = elapsed
                self.times[name]["counter"] = self.times[name]["counter"] + 1
                self.is_on[name] = False
            else:
                self.times[name]["time"] = mpi4py.MPI.Wtime()
                self.is_on[name] = True

    def get_totals(self, name: str):
        """
        Accumulated statistics for the given operation name
        This includes: is the timer still running, the total elapsed time,
        the total hit-count of the timer as well as the total time
        """
        return self.times[name]


def write_to_json(
    time_step: int,
    backend: str,
    experiment_name: str,
    init_times: List[float],
    total_times: List[float],
    main_times: List[float],
):
    """
    Given input times this function writes a json file with statistics for
    the elapsed times and the experimental setup
    """
    now = datetime.now()
    sha = git.Repo(
        pathlib.Path(__file__).parent.absolute(), search_parent_directories=True
    ).head.object.hexsha
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    filename = now.strftime("%Y-%m-%d-%H-%M-%S")
    experiment = {}
    experiment["setup"] = {}
    experiment["setup"]["experiment time"] = dt_string
    experiment["setup"]["data set"] = experiment_name
    experiment["setup"]["timesteps"] = time_step
    experiment["setup"]["hash"] = sha
    experiment["setup"]["version"] = "python/" + backend

    experiment["times"] = {}
    experiment["times"]["total"] = {}
    experiment["times"]["total"]["minimum"] = min(total_times)
    experiment["times"]["total"]["maximum"] = max(total_times)
    experiment["times"]["total"]["median"] = median(total_times)
    experiment["times"]["total"]["mean"] = mean(total_times)
    experiment["times"]["init"] = {}
    experiment["times"]["init"]["minimum"] = min(init_times)
    experiment["times"]["init"]["maximum"] = max(init_times)
    experiment["times"]["init"]["median"] = median(init_times)
    experiment["times"]["init"]["mean"] = mean(init_times)
    experiment["times"]["main"] = {}
    experiment["times"]["main"]["minimum"] = min(main_times)
    experiment["times"]["main"]["maximum"] = max(main_times)
    experiment["times"]["main"]["median"] = median(main_times)
    experiment["times"]["main"]["mean"] = mean(main_times)
    experiment["times"]["cleanup"] = {}

    with open(filename + ".json", "w") as outfile:
        json.dump(experiment, outfile, sort_keys=True, indent=4)
