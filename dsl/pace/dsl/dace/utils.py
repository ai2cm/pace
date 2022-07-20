from dataclasses import dataclass, field
import time

from dace import SDFG, SDFGState, nodes, StorageType
from pace.dsl.dace.dace_config import DaceConfig
from typing import Dict, List


class DaCeProgress:
    """Timer and log to track build progress"""

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


def is_ref(sd: SDFG, aname: str):
    found = False
    for node, state in sd.all_nodes_recursive():
        if not isinstance(state, SDFGState):
            continue
        if state.parent is sd:
            if isinstance(node, nodes.AccessNode) and aname == node.data:
                found = True
                break

    return found


@dataclass
class ArrayReport:
    name: str = ""
    total_size_in_bytes: int = 0
    referenced: bool = False
    transient: bool = False
    top_level: bool = False


@dataclass
class StorageReport:
    name: str = ""
    referenced_in_bytes: int = 0
    unreferenced_in_bytes: int = 0
    top_level_in_bytes: int = 0
    details: List[ArrayReport] = field(default_factory=list)


def count_memory(sdfg: SDFG, detail_report=False) -> str:
    allocations: Dict[StorageReport] = {}
    for storage_type in StorageType:
        allocations[storage_type] = StorageReport(name=storage_type)

    for sd, aname, arr in sdfg.arrays_recursive():
        array_size_in_bytes = arr.total_size * arr.dtype.bytes
        ref = is_ref(sd, aname)

        if sd is not sdfg and arr.transient:
            allocations[arr.storage].details.append(
                ArrayReport(
                    name=aname,
                    total_size_in_bytes=array_size_in_bytes,
                    referenced=ref,
                    transient=arr.transient,
                    top_level=False,
                )
            )
            if ref:
                allocations[arr.storage].referenced_in_bytes += array_size_in_bytes
            else:
                allocations[arr.storage].unreferenced_in_bytes += array_size_in_bytes

        elif sd is sdfg:
            allocations[arr.storage].details.append(
                ArrayReport(
                    name=aname,
                    total_size_in_bytes=array_size_in_bytes,
                    referenced=ref,
                    transient=arr.transient,
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
        toplvlalloc_in_mb = float(allocs.top_level_in_bytes / (1024 * 1024))
        if alloc_in_mb or toplvlalloc_in_mb > 0:
            report += (
                f"{storage}:\n"
                f"  Alloc ref | unref: {alloc_in_mb:.2f}mb | {unref_alloc_in_mb:.2f}mb\n"
                f"  Top lvl alloc: {toplvlalloc_in_mb:.2f}mb\n"
            )
            if detail_report:
                report += "\n"
                report += "  Referenced\tTransient\tTotal size(mb)\tName\n"
                for detail in allocs.details:
                    size_in_mb = float(detail.total_size_in_bytes / (1024 * 1024))
                    ref_str = "     X     " if detail.referenced else "           "
                    transient_str = "     X     " if detail.transient else "           "
                    report += f" {ref_str}\t{transient_str}\t   {size_in_mb:.2f}   \t{detail.name}\n"

    return report


def count_memory_from_path(sdfg_path: str, detail_report=False) -> str:
    return count_memory(SDFG.from_file(sdfg_path), detail_report=detail_report)
