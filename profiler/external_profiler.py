"""Adding semantic marking to external profiler.

Usage: python external_profiler.py <PYTHON SCRIPT>.py <ARGS>

Works with nvtx (via cupy) for now.
"""

import sys
from argparse import ArgumentParser

from tools import nvtx_markings, stencil_reproducer


try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


def parse_args():
    usage = "usage: python %(prog)s <--nvtx> <--stencil=STENCIL_NAME> <CMD TO PROFILE>"
    parser = ArgumentParser(usage=usage)
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="enable NVTX marking",
    )
    parser.add_argument(
        "--stencil",
        type=str,
        action="store",
        help="create a small reproducer for the stencil",
    )
    return parser.parse_known_args()


def profile_hook(frame, event, args):
    if cmd_line_args.nvtx and nvtx_markings.mark is not None:
        nvtx_markings.mark(frame, event, args)
    if cmd_line_args.stencil and stencil_reproducer.field_serialization is not None:
        stencil_reproducer.field_serialization(frame, event, args)


cmd_line_args = None
if __name__ == "__main__":
    cmd_line_args, unknown = parse_args()
    print(f"{cmd_line_args}")
    print(f"{unknown}")
    if cmd_line_args.nvtx and cp is None:
        print("WARNING: cupy isn't available, NVTX marking deactivated.")
        cmd_line_args.nvtx = False
    if cmd_line_args.stencil is not None:
        stencil_reproducer.collect_stencil_candidate(cmd_line_args.stencil)
    if cmd_line_args.nvtx or cmd_line_args.stencil:
        sys.setprofile(profile_hook)
    filename = unknown[0]
    sys.argv = unknown[0:]
    exec(compile(open(filename, "rb").read(), filename, "exec"))
