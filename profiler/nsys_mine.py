"""Mine the sqlite database for information on the CUDA kernels

See arg_parse() for usage
"""

import csv
import os
import re
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plot
import numpy as np
from nsys_data_mining.kernelquery import CUDAKernelTrace, KernelReportIndexing
from nsys_data_mining.nvtxquery import CUDANVTXTrace, NVTXReportIndexing
from tabulate import tabulate


# TODO: All of those should be part of a .json if we move it out of `fv3core`
FV3_MAINLOOP = "step_dynamics"
FV3_STAGES = [
    "Acoustic timestep",
    "Tracer advection",
]  # TODO remap is not tagged in nvtx

FV3_START_ASYNC_HALOS = [
    "HaloEx: async scalar",
    "HaloEx: async vector",
]

FV3_ASYNC_HALOS = FV3_START_ASYNC_HALOS + [
    "HaloEx: unpack and wait",
]

FV3_NOT_HALOS = ["Pre HaloEx"]


def _count_calls_start_with_name(rows: List[str], index: int, target_name: str) -> int:
    hit = 0
    for row in rows:
        if row[index].startswith(target_name):
            hit += 1
    return hit


def _filter_cupy_out(rows: List[str], index: int) -> List[str]:
    hit = []
    for row in rows:
        if row[index].startswith("cupy") or row[index].startswith("neg_f64"):
            continue
        hit.append(row)
    return hit


def _plot_total_time(fv3_kernel_timings: Dict[str, List[int]]):
    fig, ax = plot.subplots()
    fig.set_figwidth(30)
    fig.set_figheight(10)
    kernel_names = []
    kernel_total_time = []
    for key, timings in fv3_kernel_timings.items():
        kernel_names.append(key)
        total_time = 0
        for v in timings:
            total_time += v
        kernel_total_time.append(v)
    plot.xticks(rotation=45, ha="right")
    ax.bar(kernel_names, kernel_total_time, align="edge", color="tab:blue")
    ax.set_xlabel("Kernels")
    ax.set_ylabel("Total time in ns")
    plot.tight_layout()
    plot.savefig("total_times_per_kernel.png")


def _plot_total_time_and_counts(fv3_kernel_timings: Dict[str, List[int]]):
    fig, ax = plot.subplots()
    fig.set_figwidth(30)
    fig.set_figheight(10)
    kernel_names = []
    kernel_total_time = []
    kernel_count = []
    for key, timings in fv3_kernel_timings.items():
        kernel_names.append(key)
        total_time = 0
        hits = len(timings)
        for v in timings:
            total_time += v
        kernel_total_time.append(total_time)
        kernel_count.append(hits)

    w = 0.3
    plot.xticks(rotation=45, ha="right")
    ax.bar(kernel_names, kernel_total_time, width=-w, align="edge", color="tab:blue")
    ax_counts = ax.twinx()
    ax_counts.bar(kernel_names, kernel_count, width=w, align="edge", color="tab:orange")
    ax.set_xlabel("Kernels")
    ax.set_ylabel("Total time in ns")
    ax_counts.set_ylabel("# calls")
    plot.tight_layout()
    plot.savefig("total_times_counts_per_kernel.png")


def _print_total_time_kernel_table(
    fv3_kernel_timings: Dict[str, List[int]],
    timestep_time_in_ms: float,
    write_csv: Optional[bool] = False,
):
    kernels = []
    for name, timings in fv3_kernel_timings.items():
        total_time = 0
        hits = len(timings)
        for v in timings:
            total_time += v
        percent_of_total_in_ms = ((total_time / 1e6) / timestep_time_in_ms) * 100
        kernels.append([name, total_time, percent_of_total_in_ms, hits])
    kernels.sort(key=lambda x: x[1], reverse=True)  # type: ignore
    print("Kernel time - sorted by cumulative time")
    table = tabulate(
        kernels,
        headers=["Name", "Time", "%% of timestep", "Count"],
        tablefmt="orgtbl",
    )
    print(f"{table}")
    if write_csv:
        with open("total_time_kernel.csv", "w") as csvfile:
            csv.writer(csvfile).writerows(kernels)


def _plot_median_time(fv3_kernel_timings: Dict[str, List[int]]):
    fig, ax = plot.subplots()
    fig.set_figwidth(30)
    fig.set_figheight(10)
    kernel_names = []
    kernel_median_time = []
    for name, timings in fv3_kernel_timings.items():
        kernel_names.append(name)
        kernel_median_time.append(np.median(timings))

    ax.bar(kernel_names, kernel_median_time)
    plot.xticks(rotation=45, ha="right")
    plot.xlabel("Kernels")
    plot.ylabel("Median in ns")
    plot.tight_layout()
    plot.savefig("median_time_per_kernel.png")


def _print_median_time_kernel_table(
    fv3_kernel_timings: Dict[str, List[int]],
    write_csv: Optional[bool] = False,
):
    kernels = []
    for name, timings in fv3_kernel_timings.items():
        hits = len(timings)
        median = np.median(timings)
        kernels.append([name, median, hits])
    kernels.sort(key=lambda x: x[1], reverse=True)
    table = tabulate(kernels, headers=["Name", "Time", "Count"], tablefmt="orgtbl")
    print(f"{table}")
    if write_csv:
        with open("median_time_kernel.csv", "w") as csvfile:
            csv.writer(csvfile).writerows(kernels)


def _plot_total_call(fv3_kernel_timings: Dict[str, List[int]]):
    fig, ax = plot.subplots()
    fig.set_figwidth(30)
    fig.set_figheight(10)
    kernel_names = []
    kernel_total_hits = []
    for key, timings in fv3_kernel_timings.items():
        kernel_names.append(key)
        hits = len(timings)
        kernel_total_hits.append(hits)

    ax.bar(kernel_names, kernel_total_hits)
    plot.xticks(rotation=45, ha="right")
    plot.xlabel("Kernels")
    plot.ylabel("Total calls")
    plot.tight_layout()
    plot.savefig("call_per_kernel.png")


def _filter_kernel_name(kernels: List[Any]) -> List[Any]:
    """Filter the gridtools c++ kernel name to a readable name"""
    # Run a query to convert the stencil generated string to a readable one
    approx_stencil_name_re = re.search(
        "(?<=bound_functorIN)(.*)(?=___gtcuda)",
        kernels[KernelReportIndexing.NAME.value],
    )
    if approx_stencil_name_re is None:
        return kernels
    approx_stencil_name = approx_stencil_name_re.group().lstrip("0123456789 ")
    row_as_list = list(kernels)
    row_as_list[KernelReportIndexing.NAME.value] = approx_stencil_name
    return row_as_list


def parse_args():
    usage = "usage: python %(prog)s <--csv> <--plots> <database>"
    parser = ArgumentParser(usage=usage)
    parser.add_argument("database", type=str, help="sqlite or qdrep path to file")
    parser.add_argument(
        "--csv",
        action="store_true",
        help="write down CSV",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="plot the cumulative times/calls and median plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command and prepare sqlite DB
    cmd_line_args = parse_args()
    if cmd_line_args.database.endswith(".sqlite"):
        sql_db = cmd_line_args.database
    elif cmd_line_args.database.endswith(".qdrep"):
        status = os.system("which nsys")
        if status != 0:
            print(
                f"Couldn't find nsys to convert {cmd_line_args.database} to sqlite."
                " Abort."
            )
            exit()
        print(
            f"WARNING: this will now export a .sqlite DB from {cmd_line_args.database}"
            f"into the same directory.\n"
            f"The resulting file can be large.\n"
        )
        status = os.system(f"nsys export -t sqlite -f on {cmd_line_args.database}")
        if status != 0:
            print(
                f"Something went wrong when converting {cmd_line_args.database}"
                "  to sqlite. Abort."
            )
            exit()
        sql_db = cmd_line_args.database.replace(".qdrep", ".sqlite")
    else:
        raise RuntimeError("Cmd needs a '.sqlite' or '.qdrep'.")

    # Extract kernel info & nvtx tagging
    kernels_results = CUDAKernelTrace.Run(sql_db, sys.argv[:2])
    nvtx_results = CUDANVTXTrace.Run(sql_db, sys.argv[:2])

    # Grab second mainloop timings
    skip_first_mainloop = True
    min_start = 0
    max_end = sys.maxsize
    for row in nvtx_results:
        if row[NVTXReportIndexing.TEXT.value] == FV3_MAINLOOP:
            if skip_first_mainloop is True:
                skip_first_mainloop = False
            else:
                min_start = row[NVTXReportIndexing.START.value]
                max_end = row[NVTXReportIndexing.END.value]
                break
    assert skip_first_mainloop is False and min_start != 0
    timestep_time_in_ms = (float(max_end) - float(min_start)) * 1.0e3

    # Gather HaloEx markers
    filtered_halo_nvtx = []
    for row in nvtx_results:
        if (
            row[NVTXReportIndexing.TEXT.value] in FV3_ASYNC_HALOS
            and min_start < row[NVTXReportIndexing.START.value]
            and max_end > row[NVTXReportIndexing.END.value]
        ):
            filtered_halo_nvtx.append(row)
    # > Compute total halo time (including waiting for previous work to finish)
    total_halo_ex = 0
    for row in filtered_halo_nvtx:
        total_halo_ex += row[NVTXReportIndexing.DURATION.value]
    # > Substract the "non halo work" done under halo markings
    total_non_halo_ex = 0
    for row in nvtx_results:
        if (
            row[NVTXReportIndexing.TEXT.value] in FV3_NOT_HALOS
            and min_start < row[NVTXReportIndexing.START.value]
            and max_end > row[NVTXReportIndexing.END.value]
        ):
            total_non_halo_ex += row[NVTXReportIndexing.DURATION.value]
    if total_halo_ex != 0 and total_non_halo_ex != 0:
        halo_ex_time_in_ms = (total_halo_ex - total_non_halo_ex) / 1e6
        # > Count all halos
        only_start_halo = 0
        for row in filtered_halo_nvtx:
            if row[NVTXReportIndexing.TEXT.value] in FV3_START_ASYNC_HALOS:
                only_start_halo += 1
    else:
        # > No halo nvtx tags - can't calculate
        halo_ex_time_in_ms = "Could not calculate"  # type: ignore
        only_start_halo = "Could not calculate"  # type: ignore

    # Filter the rows between - min_start/max_end
    filtered_rows = []
    for row in kernels_results:
        if row is None:
            continue
        row = _filter_kernel_name(row)
        if (
            row[KernelReportIndexing.START.value] > min_start
            and row[KernelReportIndexing.END.value] < max_end
        ):
            filtered_rows.append(row)

    # Split results between CUPY & FV3
    all_kernels_count = len(filtered_rows)
    cupy_copies_kernels_count = _count_calls_start_with_name(
        filtered_rows, KernelReportIndexing.NAME.value, "cupy"
    )
    cupy_copies_kernels_count += _count_calls_start_with_name(
        filtered_rows, KernelReportIndexing.NAME.value, "neg_f64"
    )
    fv3_kernels_count = all_kernels_count - cupy_copies_kernels_count
    fv3_kernels = _filter_cupy_out(filtered_rows, KernelReportIndexing.NAME.value)
    assert fv3_kernels_count == len(fv3_kernels)

    # Count unique kernel
    unique_fv3_kernel_timings: Dict[str, List[Any]] = {}
    for row in fv3_kernels:
        name = row[KernelReportIndexing.NAME.value]
        if name not in unique_fv3_kernel_timings.keys():
            unique_fv3_kernel_timings[name] = []
        unique_fv3_kernel_timings[name].append(row[KernelReportIndexing.DURATION.value])

    # Plot results
    if cmd_line_args.plots:
        _plot_total_time_and_counts(unique_fv3_kernel_timings)
        _plot_total_time(unique_fv3_kernel_timings)
        _plot_total_call(unique_fv3_kernel_timings)
        _plot_median_time(unique_fv3_kernel_timings)

    # Print results
    _print_total_time_kernel_table(
        unique_fv3_kernel_timings, timestep_time_in_ms, write_csv=cmd_line_args.csv
    )
    _print_median_time_kernel_table(
        unique_fv3_kernel_timings, write_csv=cmd_line_args.csv
    )

    # Final summary print
    print(
        f"==== SUMMARY ====\n"
        f"Timestep time in ms: {timestep_time_in_ms:.2f}\n"
        f"Unique kernels: {len(unique_fv3_kernel_timings)}\n"
        f"CUDA Kernel calls:\n"
        f"  FV3 {fv3_kernels_count}/{all_kernels_count}\n"
        f"  CUPY {cupy_copies_kernels_count}/{all_kernels_count}\n"
        f"Halo exchange:\n"
        f"  count: {only_start_halo}\n"
        f"  cumulative time: {halo_ex_time_in_ms:.2f}ms "
        f"({( halo_ex_time_in_ms / timestep_time_in_ms )*100:.2f}%)\n"
    )
