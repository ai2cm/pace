import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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


def sdfg_nan_checker(sdfg: dace.SDFG):
    """
    Insert a check on array after each computational map to check for NaN
    in the domain.

    In current pipeline, it is to be inserter after sdfg.simplify(...).
    """
    import copy

    import sympy as sp

    from dace import data as dt
    from dace import symbolic
    from dace.sdfg import graph as gr
    from dace.sdfg import utils as sdutil

    # Adds a NaN checker after every mapexit->access node
    checks: List[
        Tuple[dace.SDFGState, dace.nodes.AccessNode, gr.MultiConnectorEdge[dace.Memlet]]
    ] = []
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, dace.nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]
    for me, state in topmaps:
        mx = state.exit_node(me)
        for e in state.out_edges(mx):
            if isinstance(e.dst, dace.nodes.AccessNode):
                if isinstance(e.dst.desc(state.parent), dt.View):  # Skip views for now
                    continue
                node = sdutil.get_last_view_node(state, e.dst)
                if "diss_estd" in node.data:
                    continue
                if state.memlet_path(e)[
                    0
                ].data.dynamic:  # Skip dynamic (region) outputs
                    continue

                checks.append((state, node, e))
    for state, node, e in checks:
        # Append node that will go after the map
        newnode: dace.nodes.AccessNode = copy.deepcopy(node)
        # Move all outgoing edges to new node
        for oe in list(state.out_edges(node)):
            state.remove_edge(oe)
            state.add_edge(newnode, oe.src_conn, oe.dst, oe.dst_conn, oe.data)

        # Add map in between node and newnode
        sdfg = state.parent
        inparr = sdfg.arrays[newnode.data]
        index_expr = ", ".join(["__i%d" % i for i in range(len(inparr.shape))])
        index_printf = ", ".join(["%d"] * len(inparr.shape))

        # Get range from memlet (which may not be the entire array size)
        def evaluate(expr):
            return expr.subs({sp.Function("int_floor"): symbolic.int_floor})

        # Infer scheduly
        schedule_type = dace.ScheduleType.Default
        if (
            inparr.storage == dace.StorageType.GPU_Global
            or inparr.storage == dace.StorageType.GPU_Shared
        ):
            schedule_type = dace.ScheduleType.GPU_Device

        ranges = []
        for i, (begin, end, step) in enumerate(e.data.subset):
            ranges.append(
                (f"__i{i}", (evaluate(begin), evaluate(end), evaluate(step)))
            )  # evaluate to resolve views & actively read/write domains
        state.add_mapped_tasklet(
            name="nancheck",
            map_ranges=ranges,
            inputs={"__inp": dace.Memlet.simple(newnode.data, index_expr)},
            code=f"""
            if (__inp != __inp) {{
                printf("NaN value found at {newnode.data}, line %d, index {index_printf}\\n", __LINE__, {index_expr});
            }}
            """,  # noqa: E501
            schedule=schedule_type,
            language=dace.Language.CPP,
            outputs={
                "__out": dace.Memlet.simple(newnode.data, index_expr, num_accesses=-1)
            },
            input_nodes={node.data: node},
            output_nodes={newnode.data: newnode},
            external_edges=True,
        )
    logger.info(f"Added {len(checks)} NaN checks")


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
