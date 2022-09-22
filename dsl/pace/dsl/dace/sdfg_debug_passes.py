import copy
import logging
from typing import List, Tuple

import dace
import sympy as sp
from dace import data as dt
from dace import symbolic
from dace.sdfg import graph as gr
from dace.sdfg import utils as sdutil
from dace.transformation.helpers import get_parent_map


logger = logging.getLogger(__name__)


def _filter_all_maps(
    sdfg: dace.SDFG,
    whitelist: List[str] = None,
    blacklist: List[str] = None,
    skyp_dynamic_memlet=True,
) -> List[
    Tuple[dace.SDFGState, dace.nodes.AccessNode, gr.MultiConnectorEdge[dace.Memlet]]
]:
    """
    Grab all maps outputs and filter by variable name (either black or whitelist)

    Arguments:
        sdfg: SDFG to be read. Read-only
        whitelist: filter out everyvariable NOT in this list
        blacklist: filter out every variable in this list
        skyp_dynamic_memlet: skip the memlet flagged as dynamic (regions)

    Return:
        A list of access nodes, with their state & edges organized as
        [state, node, edges]
    """

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
                # Whitelist
                if whitelist is not None:
                    if all([varname not in node.data for varname in whitelist]):
                        continue
                # Blacklist
                if blacklist is not None:
                    if any([varname in node.data for varname in blacklist]):
                        continue
                if (
                    skyp_dynamic_memlet and state.memlet_path(e)[0].data.dynamic
                ):  # Skip dynamic (region) outputs
                    print(f"Skip {node.data} (dynamic)")
                    continue

                checks.append((state, node, e))
    return checks


def _check_node(
    state: dace.sdfg.SDFGState,
    node: dace.nodes.Node,
    edge: dace.InterstateEdge,
    kernel_name: str,
    c_varname: str,
    check_c_code: str,
    comment_c_code: str,
    assert_out: bool = False,
):
    """
    Grab all maps outputs and filter by variable name (either black or whitelist)

    Arguments:
        state: SDFG-state to be modified in-palce.
        node: original node to insert check after
        edge: original output edge of the node
        kernel_name: kernel name for C code generation
        c_varname: variable name for C code generation (must
        be reused in check_c_code)
        check_c_code: conditional code for C code generation
        comment_c_code: pure string printed on hit for C code
        generation
        assert_out: insert an assert if the conditional is met (after
        the print). WARNING: on GPU the assert might put CUDA in an
        unrecoverable state.

    Return:
        None. The SDFG was modified in-place via the state, node, edge combo
    """

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
    for i, (begin, end, step) in enumerate(edge.data.subset):
        ranges.append(
            (f"__i{i}", (evaluate(begin), evaluate(end), evaluate(step)))
        )  # evaluate to resolve views & actively read/write domains
    state.add_mapped_tasklet(
        name=kernel_name,
        map_ranges=ranges,
        inputs={f"{c_varname}": dace.Memlet.simple(newnode.data, index_expr)},
        code=f"""
        if ({check_c_code}) {{
            printf("{node.data} value (%f) {comment_c_code} at line %d, index {index_printf}\\n", {c_varname}, __LINE__, {index_expr});
            {'assert(0);' if assert_out else ''}
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


def trace_all_outputs_at_index(sdfg: dace.SDFG, i: int, j: int, k: int):
    """
    Prints value for all variable when written for a specific index.
    """
    allmaps_filtered = _filter_all_maps(
        sdfg,
        skyp_dynamic_memlet=False,
    )

    for state, node, e in allmaps_filtered:
        _check_node(
            state,
            node,
            e,
            "tracer_outputs",
            "_inp",
            f"__i0 == {i} && __i1 == {j} && __i2 == {k}",
            "",
            assert_out=False,
        )

    logger.info(f"Added {len(allmaps_filtered)} ouputs trace at {i},{j},{k}")


def negative_delp_checker(sdfg: dace.SDFG):
    """
    Adds a negative check on every variable name containing "delp" when
    written to. Assert when check is True.
    """
    allmaps_filtered = _filter_all_maps(
        sdfg,
        whitelist=["delp"],
        skyp_dynamic_memlet=False,
    )

    for state, node, e in allmaps_filtered:
        _check_node(
            state,
            node,
            e,
            "neg_delp_check",
            "_inp",
            "_inp < 0",
            "delp* is negative",
            assert_out=True,
        )

    logger.info(f"Added {len(allmaps_filtered)} delp* < 0 checks")


def negative_qtracers_checker(sdfg: dace.SDFG):
    """
    Adds a negative check on every tracers via their name when
    written to. Assert when check is True.
    """
    allmaps_filtered = _filter_all_maps(
        sdfg,
        whitelist=[
            "qvapor",
            "qliquid",
            "qrain",
            "qice",
            "qsnow",
            "qgraupel",
            "qo3mr",
            "qsgs_tke",
            "qcld",
        ],
        skyp_dynamic_memlet=False,
    )

    for state, node, e in allmaps_filtered:
        _check_node(
            state,
            node,
            e,
            "neg_tracers_check",
            "_inp",
            "_inp < -1e-8",
            "tracer is negative",
            assert_out=True,
        )

    logger.info(f"Added {len(allmaps_filtered)} tracer < 0 checks")


def sdfg_nan_checker(sdfg: dace.SDFG):
    """
    Insert a check on array after each computational map to check for NaN
    in the domain. Assert when check is True.
    """
    allmaps_filtered = _filter_all_maps(
        sdfg,
        blacklist=["diss_estd"],
    )

    for state, node, e in allmaps_filtered:
        _check_node(
            state,
            node,
            e,
            "nan_check",
            "_inp",
            "_inp != _inp",
            "NaN found",
            assert_out=True,
        )

    logger.info(f"Added {len(allmaps_filtered)} NaN checks")
