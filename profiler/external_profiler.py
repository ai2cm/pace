"""Adding semantic marking to external profiler

Usage: python external_profiler.py <PYTHON SCRIPT>.py <ARGS>

Works with nvtx (via cupy) for now.
"""

import sys


try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


def get_stencil_name(frame, event, args) -> str:
    """Get the name of the stencil from within a call to FrozenStencil.__call__"""
    name = getattr(
        frame.f_locals["self"].stencil_object,
        "__name__",
        repr(frame.f_locals["self"].stencil_object.options["name"]),
    )
    return f"{name}.__call__"


def get_name_from_frame(frame, event, args) -> str:
    """Static name from frame object"""
    return frame.f_code.co_name


""" List of hook descriptors

Each entry define a unique id (function name + filename[Optional]) and a function
that gives back a str for the marker.

TODO: this is a poor-person JSON, a ppjson if you will, it could be extracted as an
configuration file if there's a usage for it
"""
functions_desc = [
    {
        "fn": "__call__",
        "file": "fv3core/decorators.py",
        "name": get_stencil_name,
    },  # All call from StencilX decorators
    {
        "fn": "__call__",
        "file": "fv3core/stencils/dyn_core.py",
        "name": "Acoustic timestep",
    },
    {
        "fn": "__call__",
        "file": "fv3core/stencils/tracer_2d_1l.py",
        "name": "Tracer advection",
    },
    {"fn": "compute", "file": "fv3core/stencils/remapping.py", "name": "Remapping"},
    {
        "fn": "step_dynamics",
        "file": "fv3core/stencils/fv_dynamics.py",
        "name": get_name_from_frame,
    },
    {
        "fn": "halo_update",
        "file": None,
        "name": "HaloEx: sync scalar",
    },  # Synchroneous halo update
    {
        "fn": "vector_halo_update",
        "file": None,
        "name": "HaloEx: sync vector",
    },  # Synchroneous vector halo update
    {
        "fn": "start_halo_update",
        "file": None,
        "name": "HaloEx: async scalar",
    },  # Asynchroneous halo update
    {
        "fn": "start_vector_halo_update",
        "file": None,
        "name": "HaloEx: async vector",
    },  # Asynchroneous vector halo update
]


def profile_hook(frame, event, args):
    """Hook at each function call & exit to record a Mark"""
    if event == "call":
        for fn_desc in functions_desc:
            if frame.f_code.co_name == fn_desc["fn"] and (
                fn_desc["file"] is None or fn_desc["file"] in frame.f_code.co_filename
            ):
                name = (
                    fn_desc["name"]
                    if isinstance(fn_desc["name"], str)
                    else fn_desc["name"](frame, event, args)
                )
                cp.cuda.nvtx.RangePush(name)
    elif event == "return":
        for fn_desc in functions_desc:
            if frame.f_code.co_name == fn_desc["fn"] and (
                fn_desc["file"] is None or fn_desc["file"] in frame.f_code.co_filename
            ):
                cp.cuda.nvtx.RangePop()


if __name__ == "__main__":
    if cp is None:
        raise RuntimeError("External profiling requires CUPY")
    sys.setprofile(profile_hook)
    filename = sys.argv[1]
    sys.argv = sys.argv[1:]
    exec(compile(open(filename, "rb").read(), filename, "exec"))
