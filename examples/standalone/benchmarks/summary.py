#!/usr/bin/env python3

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict

import numpy as np


def parse_arguments():
    """parses the arguments of the program for the data directory"""
    usage = "python %(prog)s <data_dir>"
    parser = ArgumentParser(usage=usage)
    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing potential subdirectories with "
        "the json files with performance data",
    )
    return parser.parse_args()


def get_statistics_from_data(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    collects basic statistics from the raw data and writes them into the summary file
    """

    median_points = []
    mean_points = []
    for rankdata in data_dict["times"]:
        median_points.append(median(rankdata))
        mean_points.append(mean(rankdata))

    returnvalue: Dict[str, Any] = {"medians": {}, "means": {}}
    returnvalue["hits"] = data_dict["hits"]

    returnvalue["medians"]["mean_of_medians"] = mean(median_points)
    returnvalue["medians"]["max_of_medians"] = max(median_points)
    returnvalue["medians"]["min_of_medians"] = min(median_points)
    returnvalue["means"]["mean_of_means"] = mean(mean_points)
    returnvalue["means"]["80th percentile of means"] = np.percentile(mean_points, 80)
    returnvalue["means"]["20th percentile of means"] = np.percentile(mean_points, 20)

    return returnvalue


def write_summary_file(fullpath: str, summary_data: Dict[str, Any]):
    """writes the summary file to the given path"""
    with open(str(Path(fullpath).with_suffix("")) + "_summary.json", "w") as f:
        json.dump(summary_data, f, sort_keys=True, indent=4)


def analyze_file_at_path(fullpath: str) -> None:
    """analyzes the file at a given path and writes a summary file"""
    with open(fullpath) as f:
        data = json.load(f)
        summary_data: Dict[str, Any] = {"times": {}, "setup": {}}
        summary_data["setup"].update(data["setup"])
        for data_set, times in data["times"].items():
            summary_data["times"][data_set] = get_statistics_from_data(times)

        write_summary_file(fullpath, summary_data)


def is_valid_file(fullpath: str):
    """checks if the file needs to be summarized"""
    return (
        fullpath.endswith(".json")
        and "memory_usage" not in fullpath
        and "summary" not in fullpath
    )


if __name__ == "__main__":
    args = parse_arguments()
    for subdir, _, files in os.walk(args.data_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            if is_valid_file(fullpath):
                analyze_file_at_path(fullpath)
        break  # We only want to look at top directory
