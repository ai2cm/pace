#!/usr/bin/env python3

import json
import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    usage = "usage: python %(prog)s <data_dir>"
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
    plots = config["plots"]

    # collect and sort the data
    alldata = []
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            if fullpath.endswith(".json"):
                with open(fullpath) as f:
                    data = json.load(f)
                    if filters in data["setup"]["dataset"]:
                        alldata.append(data)
    alldata.sort(
        key=lambda k: datetime.strptime(k["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S")
    )

    for plot_name, plot_config in plots.items():
        matplotlib.rcParams.update({"font.size": fontsize})
        plt.figure()
        for backend in plot_config["backends"]:
            backend_config = backends[backend]
            specific = [x for x in alldata if x["setup"]["version"] == backend]
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
                                elememt["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                            )
                            for elememt in specific
                        ],
                        [
                            elememt["times"][timer["name"]]["mean"]
                            / (
                                (elememt["setup"]["timesteps"] - 1)
                                if plot_config["type"] == "per_timestep"
                                else 1
                            )
                            for elememt in specific
                        ],
                        timer["linestyle"],
                        markersize=markersize,
                        label=label,
                        color=backend_config["color"],
                    )
                    if plot_variance:
                        plt.fill_between(
                            [
                                datetime.strptime(
                                    elememt["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                                )
                                for elememt in specific
                            ],
                            [
                                elememt["times"][timer["name"]]["maximum"]
                                / (
                                    (elememt["setup"]["timesteps"] - 1)
                                    if plot_config["type"] == "per_timestep"
                                    else 1
                                )
                                for elememt in specific
                            ],
                            [
                                elememt["times"][timer["name"]]["minimum"]
                                / (
                                    (elememt["setup"]["timesteps"] - 1)
                                    if plot_config["type"] == "per_timestep"
                                    else 1
                                )
                                for elememt in specific
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
