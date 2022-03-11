import os


os.environ.setdefault(
    "DACE_CONFIG",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "dace.conf"),
)
