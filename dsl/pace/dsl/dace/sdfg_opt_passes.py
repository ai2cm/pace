import dace
from typing import List, Dict
from dace import subsets
from dace import data
from dace.sdfg import graph
import collections

def strip_unused_global_in_compute_x_flux(sdfg: dace.SDFG):
    """Remove compute_x_flux al & ar variables transient representations that are
    considered as GPU_Global when they are actually transients to the Tasklet
    """
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode) and (
            "al__" in node.data or "ar__" in node.data
        ):
            for e in state.all_edges(node):
                tasklet = None
                if isinstance(state.memlet_path(e)[0].src, dace.nodes.Tasklet):
                    conn = state.memlet_path(e)[0].src_conn
                    tasklet = state.memlet_path(e)[0].src
                elif isinstance(state.memlet_path(e)[-1].dst, dace.nodes.Tasklet):
                     conn = state.memlet_path(e)[-1].dst_conn
                     tasklet = state.memlet_path(e)[-1].dst
                if tasklet is not None:
                     code_str = tasklet.code.as_string
                     dtype = state.parent.arrays[e.data.data].dtype
                     code_str = f"{conn}: dace.{dtype.to_string()}\n" + code_str
                     tasklet.code.as_string = code_str
                state.remove_memlet_path(e, True)


def refine_arrays(sdfg: dace.SDFG):
    """
    Insert after sdfg.simplify(...)
    """
    refined = 0

    # Collect all nodes that appear exactly twice: before and after a nested SDFG, and nowhere else
    names: Dict[str, List[dace.nodes.AccessNode]] = collections.defaultdict(list)
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            names[node.data].append((node, state))
    names = {k: v for k, v in names.items() if len(v) == 2}

    # Then, for all nested SDFG nodes, find those which appear in "names" and are around that SDFG
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
                        me: dace.nodes.MapEntry = nstate.entry_node(mx)
                        break
                else:
                    if me is None:
                        continue
                    raise TypeError(f"what {iname} {e.data.data}")

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

    print("Refined:", refined)


from dace.sdfg.analysis import scalar_to_symbol as s2s


def simple_cprop(sdfg: dace.SDFG):
    from dace import symbolic

    while True:
        var_to_edge: Dict[str, graph.Edge[dace.InterstateEdge]] = {}
        skip = set()
        for ise in sdfg.edges():
            for aname in ise.data.assignments.keys():
                # Appears more than once? Skip
                if aname in var_to_edge:
                    skip.add(aname)
                else:
                    var_to_edge[aname] = ise

        # Replace as necessary
        repldict = {}
        for var, ise in var_to_edge.items():
            if var in skip:
                continue
            # If depends on non-global values, skip
            fsyms = symbolic.free_symbols_and_functions(ise.data.assignments[var])
            if len(fsyms - sdfg.symbols.keys()) > 0:
                continue
            repldict[var] = ise.data.assignments[var]
            del ise.data.assignments[var]
            if var in sdfg.symbols:
                del sdfg.symbols[var]
            break
        # Propagate
        for k, v in repldict.items():
            for s in sdfg.nodes():
                s.replace(k, v)
            for e in sdfg.edges():
                e.data.replace(k, v, replace_keys=False)
        if len(repldict) == 0:
            # No more replacements done
            break
    s2s.remove_symbol_indirection(sdfg)


def splittable_region_expansion(sdfg: dace.SDFG):
    """
    Set certain StencilComputation library nodes to expand to a different
    schedule if they contain small splittable regions.
    """
    from gtc.dace.nodes import StencilComputation

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, StencilComputation):
            if node.has_splittable_regions() and "corner" in node.label:
                node.expansion_specification = [
                    "Sections",
                    "Stages",
                    "J",
                    "I",
                    "K",
                ]
                print("Reordered schedule for", node.label)
