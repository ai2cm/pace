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

    # collect and sort the data
    alldata = []
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            if fullpath.endswith(".json"):
                with open(fullpath) as f:
                    alldata.append(json.load(f))
    alldata.sort(
        key=lambda k: datetime.strptime(
            k["setup"]["experiment time"], "%d/%m/%Y %H:%M:%S"
        )
    )
    for plottype in ["mainLoop", "initTotal"]:
        keyval = ["main_loop"] if plottype == "mainLoop" else ["init", "total"]
        plt.figure()
        for backend in ["python/gtx86", "python/numpy", "fortran", "python/gtcuda"]:
            specific = [x for x in alldata if x["setup"]["version"] == backend]
            if specific:
                for key in keyval:
                    plt.plot(
                        [
                            datetime.strptime(
                                e["setup"]["experiment time"], "%d/%m/%Y %H:%M:%S"
                            )
                            for e in specific
                        ],
                        [e["times"][key]["median"] for e in specific],
                        label=key + " " + backend,
                    )
                    plt.fill_between(
                        [
                            datetime.strptime(
                                e["setup"]["experiment time"], "%d/%m/%Y %H:%M:%S"
                            )
                            for e in specific
                        ],
                        [e["times"][key]["maximum"] for e in specific],
                        [e["times"][key]["minimum"] for e in specific],
                        alpha=0.5,
                    )

        plt.gcf().autofmt_xdate()
        plt.ylabel("Execution time")
        plt.legend()
        plt.title(plottype)
        plt.figtext(
            0.5,
            0.01,
            "data: "
            + alldata[0]["setup"]["data set"]
            + "  timesteps:"
            + str(alldata[0]["setup"]["timesteps"]),
            wrap=True,
            horizontalalignment="center",
            fontsize=12,
        )
        plt.savefig("history" + plottype + ".png")
