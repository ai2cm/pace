#!/usr/bin/env python3

import json
import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    usage = "python %(prog)s <data_dir> <config.json>"
    parser = ArgumentParser(usage=usage)
    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing potential subdirectories with "
        "the json files with performance data",
    )
    parser.add_argument(
        "config_file",
        type=str,
        action="store",
        help="JSON file containing plot configuration",
    )
    args = parser.parse_args()

    # setup what to plot
    fontsize = 10
    markersize = 4
    plot_variance = True
    with open(args.config_file, "r") as json_file:
        config = json.load(json_file)
    filters = config["filters"]
    backends = config["backends"]
    timing_plots = config["timing_plots"]
    timing_bar_plots = config["timing_bar_plots"]
    memory_plots = config["memory_plots"]

    # collect and sort the data
    full_timing_data = []
    full_memory_data = []
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            if fullpath.endswith(".json"):
                with open(fullpath) as f:
                    data = json.load(f)
                    if filters in data["setup"]["dataset"]:
                        if "summary" in fullpath:
                            continue
                        elif "memory_usage" in fullpath:
                            full_memory_data.append(data)
                        else:
                            full_timing_data.append(data)

    full_timing_data.sort(
        key=lambda k: datetime.strptime(k["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S")
    )
    full_memory_data.sort(
        key=lambda k: datetime.strptime(k["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S")
    )

    for memory_plot, plot_config in memory_plots.items():
        matplotlib.rcParams.update({"font.size": fontsize})
        plt.figure()
        for backend in plot_config["backends"]:
            backend_config = backends[backend]
        specific = [x for x in full_memory_data if x["setup"]["version"] == backend]
        if plot_config["only_recent"]:
            select = []
            for datum in specific:
                tdelta = datetime.now() - datetime.strptime(
                    datum["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                )
                if tdelta.days < 7:
                    select.append(datum)
            specific = select
        if specific:
            label = backend_config["short_name"]
            plt.plot(
                [
                    datetime.strptime(
                        element["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                    )
                    for element in specific
                ],
                [element["mean"] for element in specific],
                "-o",
                markersize=markersize,
                label=label,
                color=backend_config["color"],
            )
        ax = plt.gca()
        plt.gcf().autofmt_xdate(rotation=45, ha="right")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
        plt.xticks(fontsize=fontsize)
        plt.ylabel(plot_config["y_axis_label"])
        plt.xlabel(plot_config["x_axis_label"])
        plt.yticks(fontsize=fontsize)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
            fontsize=fontsize * 0.8,
            handlelength=5,
        )
        plt.title(plot_config["title"], pad=20)
        ax.set_facecolor("white")
        plt.grid(color="silver", alpha=0.4)
        plt.gcf().set_size_inches(8, 6)
        plt.savefig("history_" + memory_plot + ".png", dpi=100)

    for plot_name, plot_config in timing_plots.items():
        matplotlib.rcParams.update({"font.size": fontsize})
        plt.figure()
        for backend in plot_config["backends"]:
            backend_config = backends[backend]
            specific = [x for x in full_timing_data if x["setup"]["version"] == backend]
            if plot_config["only_recent"]:
                select = []
                for datum in specific:
                    tdelta = datetime.now() - datetime.strptime(
                        datum["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                    )
                    if tdelta.days < 7:
                        select.append(datum)
                specific = select
            if specific:
                for timer in plot_config["timers"]:
                    label = None
                    if "mainloop" in timer["name"] or "total" in timer["name"]:
                        label = backend_config["short_name"]
                    elif "gtcuda" in backend_config["short_name"]:
                        label = backend_config["short_name"] + " " + timer["name"]
                    plt.plot(
                        [
                            datetime.strptime(
                                element["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                            )
                            for element in specific
                        ],
                        [
                            np.median(element["times"][timer["name"]]["times"])
                            for element in specific
                        ],
                        timer["linestyle"],
                        markersize=markersize,
                        label=label,
                        color=backend_config["color"],
                    )
                    if plot_config["plot_stddev"]:
                        plt.fill_between(
                            [
                                datetime.strptime(
                                    element["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                                )
                                for element in specific
                            ],
                            [
                                (
                                    np.median(element["times"][timer["name"]]["times"])
                                    + np.std(element["times"][timer["name"]]["times"])
                                )
                                for element in specific
                            ],
                            [
                                (
                                    np.median(element["times"][timer["name"]]["times"])
                                    - np.std(element["times"][timer["name"]]["times"])
                                )
                                for element in specific
                            ],
                            color=backend_config["color"],
                            alpha=0.2,
                        )
        ax = plt.gca()
        plt.gcf().autofmt_xdate(rotation=45, ha="right")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
        plt.xticks(fontsize=fontsize)
        plt.ylabel(plot_config["y_axis_label"])
        plt.xlabel(plot_config["x_axis_label"])
        plt.yticks(fontsize=fontsize)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
            fontsize=fontsize * 0.8,
            handlelength=5,
        )
        plt.title(plot_config["title"], pad=20)
        ax.set_facecolor("white")
        plt.grid(color="silver", alpha=0.4)
        plt.gcf().set_size_inches(8, 6)
        plt.savefig("history_" + plot_name + ".png", dpi=100)

    for plot_name, plot_config in timing_bar_plots.items():
        matplotlib.rcParams.update({"font.size": fontsize})
        last_fortran = [
            x for x in full_timing_data if x["setup"]["version"] == "fortran"
        ][-1]
        last_fortran_mainloop = np.median(last_fortran["times"]["mainloop"]["times"])
        for backend in plot_config["backends"]:
            plt.figure()
            # Ys are mainloop median + fortran reference
            specific = [
                x for x in full_timing_data if x["setup"]["version"] == backend
            ][-plot_config["run_to_go_back"] :]
            mainloops_speedup = [
                last_fortran_mainloop / np.median(x["times"]["mainloop"]["times"])
                for x in specific
            ]
            ys = mainloops_speedup
            # Xs are fake data + hash commits
            xs = np.arange(plot_config["run_to_go_back"])
            xs_hash = [x["setup"]["hash"][:6] for x in specific]
            # plot
            plt.bar(xs, ys)
            plt.axhline(y=1, color="r", linestyle="--")
            ax = plt.gca()
            plt.xticks(xs, xs_hash, fontsize=fontsize)
            plt.ylabel("Speed up factor")
            plt.xlabel("Commit hashes (latest to the right)")
            plt.yticks(fontsize=fontsize)
            plt.title(
                f"Speedup of {backend} vs Fortran on mainloop (last 3 runs)",
                pad=20,
            )
            ax.set_facecolor("white")
            plt.grid(color="silver", alpha=0.4)
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(f"speedup_{plot_name}_{backend.replace('/', '_')}.png", dpi=100)
