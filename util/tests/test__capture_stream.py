import ctypes
import os
import sys

import pytest

from pace.util import capture_stream


def get_libc():
    if os.uname().sysname == "Linux":
        return ctypes.cdll.LoadLibrary("libc.so.6")
    else:
        pytest.skip()


def printc(fd, text):
    libc = get_libc()
    b = bytes(text + "\n", "UTF-8")
    libc.write(fd, b, len(b))


def printpy(_, text):
    print(text)


@pytest.mark.parametrize("print_", [printc, printpy])
@pytest.mark.cpu_only
def test_capture_stream_python_print(capfdbinary, print_):
    text = "hello world"

    # This test interacts in a confusing with pytests output capturing
    # sys.stdout.fileno() is usually 1, but not here.
    fd = sys.stdout.fileno()
    with capture_stream(sys.stdout) as out:
        print_(fd, text)

    assert out.getvalue().decode("UTF-8") == text + "\n"
