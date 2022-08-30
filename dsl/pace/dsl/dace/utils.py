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


def count_memory(sdfg: dace.sdfg.SDFG, detail_report=False) -> str:
    allocations: Dict[dace.StorageType, StorageReport] = {}
    for storage_type in dace.StorageType:
        allocations[storage_type] = StorageReport(name=storage_type)

    for sd, aname, arr in sdfg.arrays_recursive():
        array_size_in_bytes = arr.total_size * arr.dtype.bytes
        ref = _is_ref(sd, aname)

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


def count_memory_from_path(sdfg_path: str, detail_report=False) -> str:
    return count_memory(dace.SDFG.from_file(sdfg_path), detail_report=detail_report)
