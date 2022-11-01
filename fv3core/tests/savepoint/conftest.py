# flake8: noqa

# This magical series of imports is to de-duplicate the conftest.py file
# between the dycore and physics tests. We can avoid this if we refactor the tests
# to all run from one directory

import pace.fv3core.testing


# this must happen before any classes from fv3core are instantiated
pace.fv3core.testing.enable_selective_validation()

import pace.stencils.testing.conftest
from pace.stencils.testing.conftest import *  # noqa: F403,F401

from . import translate


pace.stencils.testing.conftest.translate = translate  # type: ignore
