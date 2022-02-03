import pathlib
import re


PATTERN = re.compile(
    r'pyext_module = gt_utils.make_module_from_file\(\n    "[a-zA-Z0-9_\.]+", '
    r'("[a-zA-Z0-9_\.\/-]+)\/[a-zA-Z0-9_\.\/-]+.so", public_import=True'
)


def replace(match: re.Match):
    return match.group(0).replace(
        match.group(1), 'f"{pathlib.Path(__file__).parent.resolve()}'
    )


def test():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(f"{tmpdir}/.gt_cache/subdir", exist_ok=True)
        with open(f"{tmpdir}/.gt_cache/subdir/file.py", "w") as f:
            f.write(
                """

import time

import numpy as np
from numpy import dtype
from gt4py import utils as gt_utils

pyext_module = gt_utils.make_module_from_file(
    "_GT_.fv3core.stencils.divergence_damping.damping.m_damping__gtcgtcpu_ifirst_80bfbacc8e_pyext", "/Users/jeremym/python/pace/driver/examples/.gt_cache/py38_1013/gtcgtcpu_ifirst/fv3core/stencils/divergence_damping/damping/m_damping__gtcgtcpu_ifirst_80bfbacc8e_pyext.cpython-38-darwin.so", public_import=True
)


from gt4py.stencil_object import AccessKind, Boundary, DomainInfo, FieldInfo, ParameterInfo, StencilObject

"""  # noqa: E501
            )
        os.chdir(tmpdir)
        main()
        with open(".gt_cache/subdir/file.py", "r") as f:
            output = f.read()
        expected = """import pathlib


import time

import numpy as np
from numpy import dtype
from gt4py import utils as gt_utils

pyext_module = gt_utils.make_module_from_file(
    "_GT_.fv3core.stencils.divergence_damping.damping.m_damping__gtcgtcpu_ifirst_80bfbacc8e_pyext", f"{pathlib.Path(__file__).parent.resolve()}/m_damping__gtcgtcpu_ifirst_80bfbacc8e_pyext.cpython-38-darwin.so", public_import=True
)


from gt4py.stencil_object import AccessKind, Boundary, DomainInfo, FieldInfo, ParameterInfo, StencilObject

"""  # noqa: E501
        assert output == expected


def main():
    cwd = pathlib.Path()
    cache_dir = cwd / ".gt_cache"
    for path in cache_dir.rglob("**/*.py"):
        with open(path, "r") as f:
            code = f.read()
        match = PATTERN.search(code)
        if match:
            new_code = "import pathlib\n" + re.sub(
                pattern=PATTERN, repl=replace, string=code
            )
            with open(path, "w") as f:
                f.write(new_code)


if __name__ == "__main__":
    main()
