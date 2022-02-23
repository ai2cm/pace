import collections
from typing import Dict, List

import dace
from dace import data, subsets


def refine_permute_arrays(sdfg: dace.SDFG):
    """
    Insert after sdfg.simplify(...)
    """
    refined = 0
    permuted = 0

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
            cursdfg = state.parent
            for e in list(state.in_edges(node)):
                if e.data.data not in names:
                    continue
                if e.data.is_empty():
                    continue
                try:
                    oe = next(state.out_edges_by_connector(node, e.dst_conn))
                except StopIteration:
                    continue
                if state.in_degree(state.memlet_tree(e).root().edge.src) > 0:
                    continue
                if state.out_degree(state.memlet_tree(oe).root().edge.dst) > 0:
                    continue

                desc = cursdfg.arrays[e.data.data]
                if not desc.transient:
                    continue
                if desc.storage != dace.StorageType.GPU_Global:
                    continue
                if node.sdfg.number_of_nodes() > 1:
                    continue

                # Get size from map
                for an, nstate in node.sdfg.all_nodes_recursive():
                    if isinstance(an, dace.nodes.AccessNode) and an.data == e.dst_conn:
                        try:
                            mx = next(iter(nstate.predecessors(an)))
                        except StopIteration:
                            me = None
                            continue
                        me = nstate.entry_node(mx)
                        break

                if me is None:
                    continue

                # external
                state.remove_memlet_path(e)
                state.remove_memlet_path(oe)

                # internal
                nsdfg = node.sdfg
                iname = e.dst_conn
                idesc = nsdfg.arrays[iname]
                idesc.transient = True
                idesc.storage = dace.StorageType.GPU_Shared

                shp = []
                for expr in me.map.range.size()[::-1]:
                    symdict = {str(s): s for s in expr.free_symbols}
                    repldict = {}
                    if "tile_i" in symdict:
                        repldict[symdict["tile_i"]] = 0
                    if "tile_j" in symdict:
                        repldict[symdict["tile_j"]] = 0
                    shp.append(expr.subs(repldict))

                if len(shp) < 3:
                    shp += [1] * (3 - len(shp))

                idesc.shape = tuple(shp)
                idesc.strides = (shp[1], 1, 1)
                idesc.total_size = data._prod(shp)
                idesc.lifetime = dace.AllocationLifetime.Scope
                cursdfg.remove_data(e.data.data)
                refined += 1

        if (
            isinstance(node, dace.nodes.MapEntry)
            and node.map.params[-1] == "k"
            and len(node.map.params) == 3
        ):
            permuted += 1
            node.map.params = [
                node.map.params[-1],
                node.map.params[0],
                node.map.params[1],
            ]
            node.map.range = subsets.Range(
                [node.map.range[-1], node.map.range[0], node.map.range[1]]
            )

    print("Refined arrays:", refined)
    print("Permuted arrays from IJK t KIJ:", permuted)
