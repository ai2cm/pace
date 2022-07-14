import time

from dace import SDFG, SDFGState, nodes, StorageType
from pace.dsl.dace.dace_config import DaceConfig


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


def count_memory(sdfg: SDFG):
    allocations = {}
    for storage_type in StorageType:
        allocations[storage_type] = {}
        allocations[storage_type]["alloc"] = 0
        allocations[storage_type]["bad_alloc"] = 0
        allocations[storage_type]["toplvl_alloc"] = 0

    for sd, aname, arr in sdfg.arrays_recursive():
        if sd is not sdfg and arr.transient:
            if is_ref(sd, aname):
                allocations[arr.storage]["alloc"] += arr.total_size * arr.dtype.bytes
            else:
                allocations[arr.storage]["bad_alloc"] += (
                    arr.total_size * arr.dtype.bytes
                )

        elif sd is sdfg:
            allocations[arr.storage]["toplvl_alloc"] += arr.total_size * arr.dtype.bytes
            if is_ref(sd, aname):
                allocations[arr.storage]["alloc"] += arr.total_size * arr.dtype.bytes
            else:
                allocations[arr.storage]["bad_alloc"] += (
                    arr.total_size * arr.dtype.bytes
                )

    print(f"{sdfg.name}:\n")
    for storage, allocs in allocations.items():
        alloc_in_mb = int(allocs["alloc"] / (1024 * 1024))
        badalloc_in_mb = int(allocs["bad_alloc"] / (1024 * 1024))
        toplvlalloc_in_mb = int(allocs["toplvl_alloc"] / (1024 * 1024))
        if alloc_in_mb or toplvlalloc_in_mb > 0:
            print(
                f"  {storage}:\n"
                f"    Alloc ref | unref: {alloc_in_mb}mb | {badalloc_in_mb}mb\n"
                f"    Top lvl alloc: {toplvlalloc_in_mb}mb\n"
            )


def count_memory_from_path(sdfg_path: str):
    count_memory(SDFG.from_file(sdfg_path))
