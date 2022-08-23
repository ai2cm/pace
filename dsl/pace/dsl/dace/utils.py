import logging
import time
from typing import List, Tuple

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
