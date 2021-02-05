import json
import os
from argparse import ArgumentParser
from datetime import datetime

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
    args = parser.parse_args()

    # setup what to plot
    plots = {}
    plots["mainLoop"] = ["mainloop", "DynCore", "Remapping", "TracerAdvection"]
    plots["initializationTotal"] = ["initialization", "total"]
    backends = ["python/gtx86", "python/numpy", "fortran", "python/gtcuda"]

    # collect and sort the data
    alldata = []
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            if fullpath.endswith(".json"):
                with open(fullpath) as f:
                    alldata.append(json.load(f))
    alldata.sort(
        key=lambda k: datetime.strptime(k["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S")
    )

    for plottype, timers in plots.items():
        plt.figure()
        for backend in backends:
            specific = [x for x in alldata if x["setup"]["version"] == backend]
            if specific:
                for time in timers:
                    plt.plot(
                        [
                            datetime.strptime(
                                elememt["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                            )
                            for elememt in specific
                        ],
                        [
                            elememt["times"][time]["mean"]
                            / (
                                (elememt["setup"]["timesteps"] - 1)
                                if plottype == "mainLoop"
                                else 1
                            )
                            for elememt in specific
                        ],
                        "--o",
                        label=time + " " + backend,
                    )
                    plt.fill_between(
                        [
                            datetime.strptime(
                                elememt["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                            )
                            for elememt in specific
                        ],
                        [
                            elememt["times"][time]["maximum"]
                            / (
                                (elememt["setup"]["timesteps"] - 1)
                                if plottype == "mainLoop"
                                else 1
                            )
                            for elememt in specific
                        ],
                        [
                            elememt["times"][time]["minimum"]
                            / (
                                (elememt["setup"]["timesteps"] - 1)
                                if plottype == "mainLoop"
                                else 1
                            )
                            for elememt in specific
                        ],
                        alpha=0.3,
                    )

        ax = plt.axes()
        ax.set_facecolor("silver")
        plt.gcf().autofmt_xdate()
        plt.ylabel(
            "Execution time per timestep"
            if plottype == "mainLoop"
            else "Execution time"
        )
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
            fontsize=8,
        )
        plt.title(plottype, pad=20)
        plt.figtext(
            0.5,
            0.01,
            "data: "
            + alldata[0]["setup"]["dataset"]
            + "  timesteps:"
            + str(alldata[0]["setup"]["timesteps"]),
            wrap=True,
            horizontalalignment="center",
            fontsize=12,
        )
        plt.grid(color="white", alpha=0.4)
        plt.savefig("history" + plottype + ".png")
