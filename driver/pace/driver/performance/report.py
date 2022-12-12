import copy
import dataclasses
import json
from datetime import datetime
from typing import Any, Dict, List, Mapping

import numpy as np

from pace.util.comm import Comm


@dataclasses.dataclass
class Experiment:
    dataset: str
    format_version: int
    git_hash: str
    timestamp: str
    timesteps: int
    backend: str


@dataclasses.dataclass
class TimeReport:
    hits: int
    times: list


@dataclasses.dataclass
class Report:
    setup: Experiment
    times: dict
    dt_atmos: float
    sim_status: str = "Finished"
    SYPD: float = 0.0

    def __post_init__(self):
        self.SYPD = get_sypd(self.times, self.dt_atmos)


def get_experiment_info(
    experiment_name: str,
    time_step: int,
    backend: str,
    git_hash: str,
    is_orchestrated: bool,
) -> Experiment:
    orchestration = "orchestrated" if is_orchestrated else "python"
    experiment = Experiment(
        dataset=experiment_name,
        format_version=3,
        git_hash=git_hash,
        timestamp=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        timesteps=time_step,
        backend=f"{orchestration}/{backend}",
    )
    return experiment


def collect_keys_from_data(times_per_step: List[Mapping[str, float]]) -> List[str]:
    """Collects all the keys in the list of dicts and returns a sorted version"""
    keys = set()
    for data_point in times_per_step:
        for k, _ in data_point.items():
            keys.add(k)
    sorted_keys = list(keys)
    sorted_keys.sort()
    return sorted_keys


def gather_timing_data(
    times_per_step: List[Mapping[str, float]],
    comm,
    root: int = 0,
) -> Dict[str, Any]:
    """returns an updated version of the results dictionary owned
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


def write_to_timestamped_json(experiment: Report) -> None:
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d-%H-%M-%S")
    with open(filename + ".json", "w") as outfile:
        json.dump(dataclasses.asdict(experiment), outfile, sort_keys=True, indent=4)


def gather_hit_counts(
    hits_per_step: List[Mapping[str, int]], timing_info: Dict[str, TimeReport]
) -> Dict[str, TimeReport]:
    """collects the hit count across all timers called in a program execution"""
    for data_point in hits_per_step:
        for name, value in data_point.items():
            timing_info[name].hits += value
    return timing_info


def get_sypd(timing_info: Dict[str, TimeReport], dt_atmos: float) -> float:
    if "mainloop" in timing_info:
        is_list_of_list = any(
            isinstance(el, list) for el in timing_info["mainloop"].times
        )
        if is_list_of_list:
            mainloop = np.mean(sum(timing_info["mainloop"].times, []))
        else:
            mainloop = np.mean(timing_info["mainloop"].times)
        speedup = dt_atmos / mainloop
        sypd = 1.0 / 365.0 * speedup
    else:
        sypd = -999.0
    return sypd


def collect_data_and_write_to_file(
    time_step: int,
    backend: str,
    is_orchestrated: bool,
    git_hash: str,
    comm: Comm,
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
        exp_info = get_experiment_info(
            experiment_name, time_step, backend, git_hash, is_orchestrated
        )
        timing_info = gather_hit_counts(hits_per_step, timing_info)
        report = Report(setup=exp_info, times=timing_info, dt_atmos=dt_atmos)
        write_to_timestamped_json(report)
