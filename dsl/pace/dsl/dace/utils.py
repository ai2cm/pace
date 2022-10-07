import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

import dace
from dace.transformation.helpers import get_parent_map

from pace.dsl.dace.dace_config import DaceConfig


logger = logging.getLogger(__name__)

# Rough timer & log for major operations of DaCe build stack
class DaCeProgress:
    """Timer and log to track build progress"""

    def __init__(self, config: DaceConfig, label: str):
        self.prefix = DaCeProgress.default_prefix(config)
        self.prefix = f"[{config.get_orchestrate()}]"
        self.label = label

    @classmethod
    def log(cls, prefix: str, message: str):
        logger.info(f"{prefix} {message}")

    @classmethod
    def default_prefix(cls, config: DaceConfig) -> str:
        return f"[{config.get_orchestrate()}]"

    def __enter__(self):
        DaCeProgress.log(self.prefix, f"{self.label}...")
        self.start = time.time()

    def __exit__(self, _type, _val, _traceback):
        elapsed = time.time() - self.start
        DaCeProgress.log(self.prefix, f"{self.label}...{elapsed}s.")


def _is_ref(sd: dace.sdfg.SDFG, aname: str):
    found = False
    for node, state in sd.all_nodes_recursive():
        if not isinstance(state, dace.sdfg.SDFGState):
            continue
        if state.parent is sd:
            if isinstance(node, dace.nodes.AccessNode) and aname == node.data:
                found = True
                break

    return found


@dataclass
class ArrayReport:
    name: str = ""
    total_size_in_bytes: int = 0
    referenced: bool = False
    transient: bool = False
    pool: bool = False
    top_level: bool = False


@dataclass
class StorageReport:
    name: str = ""
    referenced_in_bytes: int = 0
    unreferenced_in_bytes: int = 0
    in_pooled_in_bytes: int = 0
    top_level_in_bytes: int = 0
    details: List[ArrayReport] = field(default_factory=list)


def memory_static_analysis(
    sdfg: dace.sdfg.SDFG,
) -> Dict[dace.StorageType, StorageReport]:
    """Analysis an SDFG for memory pressure.

    The results split memory by type (dace.StorageType) and account for
    allocated, unreferenced and top lovel (e.g. top-most SDFG) memory
    """
    # We report all allocation type
    allocations: Dict[dace.StorageType, StorageReport] = {}
    for storage_type in dace.StorageType:
        allocations[storage_type] = StorageReport(name=storage_type)

    for sd, aname, arr in sdfg.arrays_recursive():
        array_size_in_bytes = arr.total_size * arr.dtype.bytes
        ref = _is_ref(sd, aname)

        # Transient in maps (refrence and not referenced)
        if sd is not sdfg and arr.transient:
            if arr.pool:
                allocations[arr.storage].in_pooled_in_bytes += array_size_in_bytes
            allocations[arr.storage].details.append(
                ArrayReport(
                    name=aname,
                    total_size_in_bytes=array_size_in_bytes,
                    referenced=ref,
                    transient=arr.transient,
                    pool=arr.pool,
                    top_level=False,
                )
            )
            if ref:
                allocations[arr.storage].referenced_in_bytes += array_size_in_bytes
            else:
                allocations[arr.storage].unreferenced_in_bytes += array_size_in_bytes

        # SDFG-level memory (refrence, not referenced and pooled)
        elif sd is sdfg:
            if arr.pool:
                allocations[arr.storage].in_pooled_in_bytes += array_size_in_bytes
            allocations[arr.storage].details.append(
                ArrayReport(
                    name=aname,
                    total_size_in_bytes=array_size_in_bytes,
                    referenced=ref,
                    transient=arr.transient,
                    pool=arr.pool,
                    top_level=True,
                )
            )
            allocations[arr.storage].top_level_in_bytes += array_size_in_bytes
            if ref:
                allocations[arr.storage].referenced_in_bytes += array_size_in_bytes
            else:
                allocations[arr.storage].unreferenced_in_bytes += array_size_in_bytes

    return allocations


def report_memory_static_analysis(
    sdfg: dace.sdfg.SDFG,
    allocations: Dict[dace.StorageType, StorageReport],
    detail_report=False,
) -> str:
    """Create a human readable report form the memory analysis results"""
    report = f"{sdfg.name}:\n"
    for storage, allocs in allocations.items():
        alloc_in_mb = float(allocs.referenced_in_bytes / (1024 * 1024))
        unref_alloc_in_mb = float(allocs.unreferenced_in_bytes / (1024 * 1024))
        in_pooled_in_mb = float(allocs.in_pooled_in_bytes / (1024 * 1024))
        toplvlalloc_in_mb = float(allocs.top_level_in_bytes / (1024 * 1024))
        if alloc_in_mb or toplvlalloc_in_mb > 0:
            report += (
                f"{storage}:\n"
                f"  Alloc ref {alloc_in_mb:.2f} mb\n"
                f"  Alloc unref {unref_alloc_in_mb:.2f} mb\n"
                f"  Pooled {in_pooled_in_mb:.2f} mb\n"
                f"  Top lvl alloc: {toplvlalloc_in_mb:.2f}mb\n"
            )
            if detail_report:
                report += "\n"
                report += "  Referenced\tTransient   \tPooled\tTotal size(mb)\tName\n"
                for detail in allocs.details:
                    size_in_mb = float(detail.total_size_in_bytes / (1024 * 1024))
                    ref_str = "     X     " if detail.referenced else "           "
                    transient_str = "     X     " if detail.transient else "           "
                    pooled_str = "     X     " if detail.pool else "           "
                    report += (
                        f" {ref_str}\t{transient_str}"
                        f"\t   {pooled_str}"
                        f"\t   {size_in_mb:.2f}"
                        f"\t   {detail.name}\n"
                    )

    return report


def memory_static_analysis_from_path(sdfg_path: str, detail_report=False) -> str:
    """Open a SDFG and report the memory analysis"""
    sdfg = dace.SDFG.from_file(sdfg_path)
    return report_memory_static_analysis(
        sdfg,
        memory_static_analysis(sdfg),
        detail_report=detail_report,
    )


# TODO (floriand): in order for the timing analysis to be realistic the reference
# bandwidth of the hardware should be measured with GT4Py & simple in/out copy
# stencils. This allows to both measure the _actual_ deployed hardware and
# size it against the current GT4Py version.
# Below we bypass this needed automation by writing the P100 bw on Piz Daint
# measured with the above strategy.
# A better tool would allow this measure with a simple command and allow
# a one command that measure bw & report kernel analysis in one command
_HARDWARE_BW_GB_S = {"P100": 492.0}


def kernel_theoretical_timing(
    sdfg: dace.sdfg.SDFG, hardware="P100", hardware_bw_in_Gb_s=None
) -> Dict[str, float]:
    """Compute a lower timing bound for kernels with the following hypothesis:

    - Performance is memory bound, e.g. arithmetic intensity isn't counted
    - Hardware bandwidth comes from a GT4Py/DaCe test rather than a spec sheet for
      for higher accuracy. Best is to run a copy_stencils on a full domain
    - Memory pressure is mostly in read/write from global memory, inner scalar & shared
      memory is not counted towards memory movement.
    """
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, dace.nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]

    result: Dict[str, float] = {}
    for node, state in topmaps:
        nsdfg = state.parent
        mx = state.exit_node(node)

        # Gather all memory read & write by reading all
        # in-node & out-node memory. All in bytes
        alldata_in_bytes = sum(
            [
                dace.data._prod(e.data.subset.size())
                * nsdfg.arrays[e.data.data].dtype.bytes
                for e in state.in_edges(node)
            ]
        )
        alldata_in_bytes += sum(
            [
                dace.data._prod(e.data.subset.size())
                * nsdfg.arrays[e.data.data].dtype.bytes
                for e in state.out_edges(mx)
            ]
        )

        # Compute hardware memory bandwidth in bytes/us
        if hardware_bw_in_Gb_s and hardware in _HARDWARE_BW_GB_S.keys():
            raise NotImplementedError("can't specify hardware bandwidth and hardware")
        if hardware_bw_in_Gb_s:
            bandwidth_in_bytes_s = hardware_bw_in_Gb_s * 1024 * 1024 * 1024
        elif hardware in _HARDWARE_BW_GB_S.keys():
            # Time it has to take (at least): bytes / bandwidth_in_bytes_s
            bandwidth_in_bytes_s = _HARDWARE_BW_GB_S[hardware] * 1024 * 1024 * 1024
        else:
            print(
                f"Timing analysis: hardware {hardware} unknown and no bandwidth given"
            )

        in_us = 1000 * 1000

        # Theoretical fastest timing
        try:
            newresult_in_us = (float(alldata_in_bytes) / bandwidth_in_bytes_s) * in_us
        except TypeError:
            newresult_in_us = (alldata_in_bytes / bandwidth_in_bytes_s) * in_us

        if node.label in result:
            import sympy

            newresult_in_us = sympy.Max(result[node.label], newresult_in_us).expand()
            try:
                newresult_in_us = float(newresult_in_us)
            except TypeError:
                pass

        # Bad expansion
        if not isinstance(newresult_in_us, float):
            continue

        result[node.label] = newresult_in_us

    return result


def report_kernel_theoretical_timing(
    timings: Dict[str, float], human_readable: bool = True, csv: bool = False
) -> str:
    """Produce a human readable or CSV of the kernel timings"""
    result_string = f"Maps processed: {len(timings)}.\n"
    if human_readable:
        result_string += "Timing in microseconds  Map name:\n"
        result_string += "\n".join(f"{v:.2f}\t{k}," for k, v in sorted(timings.items()))
    if csv:
        result_string += "#Map name,timing in microseconds\n"
        result_string += "\n".join(f"{k},{v}," for k, v in sorted(timings.items()))
    return result_string


def kernel_theoretical_timing_from_path(sdfg_path: str) -> str:
    """Load an SDFG and report the theoretical kernel timings"""
    timings = kernel_theoretical_timing(dace.SDFG.from_file(sdfg_path))
    return report_kernel_theoretical_timing(timings, human_readable=True, csv=False)
