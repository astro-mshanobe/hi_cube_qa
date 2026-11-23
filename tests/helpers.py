"""Stand-ins for the pytest features the tests use, so that they also run under run_tests.py."""

import contextlib
import re
import tempfile
from pathlib import Path


def uses_tmp_path(fn):
    """Call *fn* with a fresh temporary directory (a Path) as its only argument."""

    def wrapper():
        with tempfile.TemporaryDirectory() as directory:
            return fn(Path(directory))

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@contextlib.contextmanager
def raises(exc_type, match=None):
    """Assert that the block raises *exc_type*, optionally with a message matching the regex *match*."""
    try:
        yield
    except exc_type as err:
        if match is not None and not re.search(match, str(err)):
            raise AssertionError(
                f"{exc_type.__name__} message {str(err)!r} does not match {match!r}"
            ) from err
        return
    raise AssertionError(f"{exc_type.__name__} was not raised")
