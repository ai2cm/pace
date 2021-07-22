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
        "fn": "wait",
        "file": "fv3gfs/util/halo_updater.py",
        "name": "HaloUpdater.wait",
    },
    {
        "fn": "start",
        "file": "fv3gfs/util/halo_updater.py",
        "name": "HaloUpdater.start",
    },
    {
        "fn": "async_pack",
        "file": "fv3gfs/util/halo_data_transformer.py",
        "name": "HaloDataTrf.async_pack",
    },
    {
        "fn": "async_unpack",
        "file": "fv3gfs/util/halo_data_transformer.py",
        "name": "HaloDataTrf.async_unpack",
    },
    {
        "fn": "synchronize",
        "file": "fv3gfs/util/halo_data_transformer.py",
        "name": "HaloDataTrf.synchronize",
    },
]


def mark(frame, event, args):
    """Hook at each function call & exit to record a Mark."""
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
