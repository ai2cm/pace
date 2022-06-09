# This magical series of imports is to de-duplicate the conftest.py file
# between the dycore and physics tests. We can avoid this if we refactor the tests
# to all run from one directory
import pace.stencils.testing.conftest
from pace.stencils.testing.conftest import *  # noqa: F403,F401

from . import translate


pace.stencils.testing.conftest.translate = translate
