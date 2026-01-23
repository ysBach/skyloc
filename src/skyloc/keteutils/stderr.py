# tools to redirect stderr(eprintln!) from kete's rust code.

import sys
from contextlib import contextmanager
from ._util import get_kete_loaded_objects

__all__ = [
    "filter_stderr",
    "filter_stderr_kete_loaded_asteroids",
]


class FilteredStderr:
    def __init__(self, filter_startswith, original_stderr):
        self.filter_startswith = (
            filter_startswith
            if isinstance(filter_startswith, (list, tuple))
            else [filter_startswith]
        )
        self.original_stderr = original_stderr

    def write(self, text):
        if not any(text.startswith(s) for s in self.filter_startswith):
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()


@contextmanager
def filter_stderr(filter_startswith):
    """Redirect stderr to filter out specific strings.

    Parameters
    ----------
    filter_startswith : list
        List of strings to filter out from stderr.

    Returns
    -------
    None

    Examples
    --------
    Examples
    --------
    ::

        with filter_stderr(["impact detected", "warning:"]):
            states0 = kete.propagate_n_body(
                self.states_from_orb[states_mask],
                jd=jd0,
                include_asteroids=include_asteroids,
                non_gravs=self.non_gravs,
                suppress_errors=suppress_errors,
        )
    """
    original_stderr = sys.stderr
    filtered_stderr = FilteredStderr(filter_startswith, original_stderr)
    sys.stderr = filtered_stderr
    try:
        yield
    finally:
        sys.stderr = original_stderr


def filter_stderr_kete_loaded_asteroids():
    cache = get_kete_loaded_objects()
    filter_startswith = []
    for astnum_str in cache['asteroids']:
        filter_startswith.append(f'Impact detected between (Name("{astnum_str}"))  ')
    return filter_stderr(filter_startswith)
