import time

from pace.dsl.dace.dace_config import DaceConfig


# Rough timer & log for major operations of DaCe build stack
class DaCeProgress:
    def __init__(self, config: DaceConfig, label: str):
        self.prefix = f"[{config.get_orchestrate()}]"
        self.label = label

    @classmethod
    def log(cls, prefix: str, message: str):
        print(f"{prefix} {message}")

    def __enter__(self):
        DaCeProgress.log(self.prefix, f"{self.label}...")
        self.start = time.time()

    def __exit__(self, _type, _val, _traceback):
        elapsed = time.time() - self.start
        DaCeProgress.log(self.prefix, f"{self.label}...{elapsed}s.")


import dace
from typing import List, Tuple
from dace.transformation.helpers import get_parent_map


def nanczek(sdfg: dace.SDFG):
    """
    Insert after sdfg.simplify(...)
    """
    import copy
    from dace.sdfg import graph as gr, utils as sdutil
    from dace import data as dt, symbolic, subsets as sbs
    import sympy as sp

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
        symbols_printf = ", ".join(
            [f"{s}=%d" for s, v in sorted(sdfg.symbols.items()) if v is dace.int32]
            + [f"{s}=%lld" for s, v in sorted(sdfg.symbols.items()) if v is dace.int64]
            + [f"{s}=%u" for s, v in sorted(sdfg.symbols.items()) if v is dace.uint32]
            + [f"{s}=%llu" for s, v in sorted(sdfg.symbols.items()) if v is dace.uint64]
        )
        symbols_expr = ", ".join(
            [f"{s}" for s, v in sorted(sdfg.symbols.items()) if v is dace.int32]
            + [f"{s}" for s, v in sorted(sdfg.symbols.items()) if v is dace.int64]
            + [f"{s}" for s, v in sorted(sdfg.symbols.items()) if v is dace.uint32]
            + [f"{s}" for s, v in sorted(sdfg.symbols.items()) if v is dace.uint64]
        )
        if not symbols_expr:
            symbols_printf = "%s"
            symbols_expr = '"N/A"'
        # Get range from memlet (which may not be the entire array size)
        def evaluate(expr):
            return expr.subs({sp.Function("int_floor"): symbolic.int_floor})

        ranges = []
        for i, (b, e, s) in enumerate(e.data.subset):
            ranges.append((f"__i{i}", (evaluate(b), evaluate(e), evaluate(s))))
        state.add_mapped_tasklet(
            name="nancheck",
            map_ranges=ranges,
            inputs={"__inp": dace.Memlet.simple(newnode.data, index_expr)},
            # printf("NaN value found at {newnode.data}, line %d, index {index_printf}. Symbols: {symbols_printf}\\n", __LINE__, {index_expr}, {symbols_expr});
            code=f"""
            if (__inp != __inp) {{
                printf("NaN value found at {newnode.data}, line %d, index {index_printf}\\n", __LINE__, {index_expr});
            }}
            """,
            schedule=dace.ScheduleType.GPU_Device,
            language=dace.Language.CPP,
            outputs={
                "__out": dace.Memlet.simple(newnode.data, index_expr, num_accesses=-1)
            },
            input_nodes={node.data: node},
            output_nodes={newnode.data: newnode},
            external_edges=True,
        )
    print(f"Added {len(checks)} NaN checks")


def _dace_inhibitor(fn):
    return fn


@_dace_inhibitor
def checks(arr, name):
    import cupy as cp

    if isinstance(arr, float) or isinstance(arr, bool) or isinstance(arr, int):
        print(f"Scalar {name}: {arr}")
    else:
        host_arr = cp.asnumpy(arr)
        print(f"{name} @ value [10, 5, 5]: {host_arr[10,5,5]}")
        print(f"{name} @ value [10, 5, 4]: {host_arr[10,5,4]}")
        print(f"{name} @ value [10, 5, 3]: {host_arr[10,5,3]}")


@_dace_inhibitor
def checks_fieldij(arr, name):
    import cupy as cp

    host_arr = cp.asnumpy(arr)
    print(f"{name} @ value [12, 5]: {host_arr[12,5]}")
    print(f"{name} @ value [11, 5]: {host_arr[11,5]}")
    print(f"{name} @ value [10, 5]: {host_arr[10,5]}")
    print(f"{name} @ value [9, 4]: {host_arr[9,5]}")
    print(f"{name} @ value [8, 4]: {host_arr[8,5]}")


@_dace_inhibitor
def checks_fieldk(arr, name):
    import cupy as cp

    host_arr = cp.asnumpy(arr)
    print(f"{name} @ value [:]: {host_arr[:]}")
