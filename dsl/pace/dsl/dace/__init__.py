import os


os.environ.setdefault(
    "DACE_CONFIG",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "dace.conf"),
)
os.environ.setdefault("DACE_execution_general_check_args", "0")
os.environ.setdefault("DACE_frontend_dont_fuse_callbacks", "1")
os.environ.setdefault("DACE_compiler_cpu_openmp_sections", "0")
os.environ.setdefault("DACE_compiler_cuda_max_concurrent_streams", "-1")
os.environ.setdefault("DACE_frontend_unroll_threshold", "0")
os.environ.setdefault("DACE_compiler_unique_functions", "none")
