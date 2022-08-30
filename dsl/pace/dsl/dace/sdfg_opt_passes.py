import logging

import dace


logger = logging.getLogger(__name__)


def splittable_region_expansion(sdfg: dace.SDFG, verbose: bool = False):
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
                if verbose:
                    logger.info(f"Reordered schedule for {node.label}")
