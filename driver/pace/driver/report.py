import copy
import dataclasses
import json
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from pace.util.mpi import MPI


@dataclasses.dataclass
class Experiment:
    dataset: str
    format_version: int
    git_hash: str
    timestamp: str
    timesteps: int
    version: str


@dataclasses.dataclass
class TimeReport:
    hits: int
    times: list


@dataclasses.dataclass
class Report:
    setup: Experiment
    times: dict
    SYPD: float


def get_experiment_info(
    experiment_name: str, time_step: int, backend: str, git_hash: str
) -> Dict[str, Any]:
    experiment = Experiment(
        dataset=experiment_name,
        format_version=3,
        git_hash=git_hash,
        timestamp=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        timesteps=time_step,
        version="python/" + backend,
    )
    return experiment


def collect_keys_from_data(times_per_step: List[Dict[str, float]]) -> List[str]:
    """Collects all the keys in the list of dics and returns a sorted version"""
    keys = set()
    for data_point in times_per_step:
        for k, _ in data_point.items():
            keys.add(k)
    sorted_keys = list(keys)
    sorted_keys.sort()
    return sorted_keys


def gather_timing_data(
    times_per_step: List[Dict[str, float]],
    comm,
    root: int = 0,
) -> Dict[str, Any]:
    """returns an updated version of  the results dictionary owned
    by the root node to hold data on the substeps as well as the main loop timers"""
    is_root = comm.Get_rank() == root
    keys = collect_keys_from_data(times_per_step)
    data: List[float] = []
    timing_info = {}
    for timer_name in keys:
        data.clear()
        for data_point in times_per_step:
            if timer_name in data_point:
                data.append(data_point[timer_name])

        sendbuf = np.array(data)
        recvbuf = None
        if is_root:
            recvbuf = np.array([data] * comm.Get_size())
        comm.Gather(sendbuf, recvbuf, root=0)
        if is_root:
            timing_info[timer_name] = TimeReport(
                hits=0, times=copy.deepcopy(recvbuf.tolist())
            )
    return timing_info


def write_to_timestamped_json(experiment: Dict[str, Any]) -> None:
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d-%H-%M-%S")
    with open(filename + ".json", "w") as outfile:
        json.dump(experiment, outfile, sort_keys=True, indent=4)


def gather_hit_counts(
    hits_per_step: List[Dict[str, int]], timing_info: Dict[str, TimeReport]
) -> Dict[str, Any]:
    """collects the hit count across all timers called in a program execution"""
    for data_point in hits_per_step:
        for name, value in data_point.items():
            timing_info[name].hits += value
    return timing_info


def get_sypd(timing_info: Dict[str, TimeReport], dt_atmos: float) -> float:
    if "mainloop" in timing_info:
        mainloop = np.mean(sum(timing_info["mainloop"].times, []))
        speedup = dt_atmos / mainloop
        sypd = 1.0 / 365.0 * speedup
    else:
        sypd = -999.0
    return sypd


def make_report(
    exp_info: Experiment,
    timing_info: Dict[str, TimeReport],
    dt_atmos: float,
):
    sypd = get_sypd(timing_info, dt_atmos)
    report = Report(setup=exp_info, times=timing_info, SYPD=sypd)
    return dataclasses.asdict(report)


def collect_data_and_write_to_file(
    time_step: int,
    backend: str,
    git_hash: str,
    comm: MPI.Comm,
    hits_per_step: List,
    times_per_step: List,
    experiment_name: str,
    dt_atmos: float,
) -> None:
    """
    collect the gathered data from all the ranks onto rank 0 and write the timing file
    """
    is_root = comm.Get_rank() == 0
    timing_info = gather_timing_data(times_per_step, comm)

    if is_root:
        exp_info = get_experiment_info(experiment_name, time_step, backend, git_hash)
        timing_info = gather_hit_counts(hits_per_step, timing_info)
        report = make_report(exp_info, timing_info, dt_atmos)
        write_to_timestamped_json(report)
