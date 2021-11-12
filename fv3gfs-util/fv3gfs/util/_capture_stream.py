import contextlib
import io
import os
import tempfile


@contextlib.contextmanager
def capture_stream(stream):

    out_stream = io.BytesIO()

    # parent process:
    # close the reading end, we won't need this
    orig_file_handle = os.dup(stream.fileno())

    with tempfile.NamedTemporaryFile() as out:
        # overwrite the streams fileno with a the pipe to be read by the forked
        # process below
        os.dup2(out.fileno(), stream.fileno())
        yield out_stream
        # restore the original file handle
        os.dup2(orig_file_handle, stream.fileno())
        # print logging info
        out.seek(0)
        out_stream.write(out.read())
