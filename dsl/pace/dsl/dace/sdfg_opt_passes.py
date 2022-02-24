import collections
from typing import Dict, List

import dace
from dace import data, subsets


def make_local_memory_transient(sdfg: dace.SDFG):
    """
    Detect memory used only locally to a nested SDFG and turns it into
    transient (GPU shared & scoped to the nested SDFG) when it has been
    declared has global.
    This will reduce memory pressure.

    As a custom optimization pass, this is to be inserted after the
    generic optimization call to sdfg.simplify(...)
    """
    refined = 0

    # Collect all nodes that appear exactly twice:
    #  before and after a nested SDFG, and nowhere else
    names: Dict[str, List[dace.nodes.AccessNode]] = collections.defaultdict(list)
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            names[node.data].append((node, state))
    names = {k: v for k, v in names.items() if len(v) == 2}

    # Then, for all nested SDFG nodes,
    # find those which appear in "names" and are around that SDFG
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.NestedSDFG):
            current_sdfg = state.parent
            for edge in list(state.in_edges(node)):
                if edge.data.data not in names:
                    continue
                if edge.data.is_empty():
                    continue
                try:
                    out_edge = next(state.out_edges_by_connector(node, edge.dst_conn))
                except StopIteration:
                    continue
                if state.in_degree(state.memlet_tree(edge).root().edge.src) > 0:
                    continue
                if state.out_degree(state.memlet_tree(out_edge).root().edge.dst) > 0:
                    continue

                desc = current_sdfg.arrays[edge.data.data]
                if not desc.transient:
                    continue
                if desc.storage != dace.StorageType.GPU_Global:
                    continue
                if node.sdfg.number_of_nodes() > 1:
                    continue

                # Get size from map
                for the_node, node_state in node.sdfg.all_nodes_recursive():
                    if (
                        isinstance(the_node, dace.nodes.AccessNode)
                        and the_node.data == edge.dst_conn
                    ):
                        try:
                            mx = next(iter(node_state.predecessors(the_node)))
                        except StopIteration:
                            me = None
                            continue
                        me = node_state.entry_node(mx)
                        break

                if me is None:
                    continue

                # Link to external memory node to be removed
                state.remove_memlet_path(edge)
                state.remove_memlet_path(out_edge)

                # Switch array to an internal shared memory access
                nsdfg = node.sdfg
                internal_name = edge.dst_conn
                internal_desc = nsdfg.arrays[internal_name]
                internal_desc.transient = True
                internal_desc.storage = dace.StorageType.GPU_Shared

                shape = []
                for expr in me.map.range.size()[::-1]:
                    symdict = {str(s): s for s in expr.free_symbols}
                    repldict = {}
                    if "tile_i" in symdict:
                        repldict[symdict["tile_i"]] = 0
                    if "tile_j" in symdict:
                        repldict[symdict["tile_j"]] = 0
                    shape.append(expr.subs(repldict))

                if len(shape) < 3:
                    shape += [1] * (3 - len(shape))

                internal_desc.shape = tuple(shape)
                internal_desc.strides = (shape[1], 1, 1)
                internal_desc.total_size = data._prod(shape)
                internal_desc.lifetime = dace.AllocationLifetime.Scope
                current_sdfg.remove_data(edge.data.data)
                refined += 1


def flip_default_layout_to_KIJ_on_maps(sdfg: dace.SDFG):
    """Flip the default layout of all maps from IJK to KIJ

    As a custom optimization pass, this is to be inserted after the
    generic optimization call to sdfg.simplify(...)
    """
    array_flipped = 0
    for node, _state in sdfg.all_nodes_recursive():

        if (
            isinstance(node, dace.nodes.MapEntry)
            and node.map.params[-1] == "k"
            and len(node.map.params) == 3
        ):
            array_flipped += 1
            node.map.params = [
                node.map.params[-1],
                node.map.params[0],
                node.map.params[1],
            ]
            node.map.range = subsets.Range(
                [node.map.range[-1], node.map.range[0], node.map.range[1]]
            )
