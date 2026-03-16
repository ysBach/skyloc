"""Stderr filtering utilities for suppressing kete's Rust-level messages.

Kete's n-body propagator prints impact-detection warnings to stderr via
Rust's ``eprintln!`` macro.  These utilities temporarily redirect
``sys.stderr`` to filter out those messages during propagation runs.
"""

# tools to redirect stderr(eprintln!) from kete's rust code.

import sys
from contextlib import contextmanager
from ._util import get_kete_loaded_objects

__all__ = [
    "filter_stderr",
    "filter_stderr_kete_loaded_asteroids",
]


class FilteredStderr:
    """File-like stderr wrapper that suppresses lines matching given prefixes.

    Parameters
    ----------
    filter_startswith : str or list of str
        Prefix(es) of stderr lines to suppress.
    original_stderr : file-like
        The original ``sys.stderr`` to delegate non-suppressed writes to.
    """

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
    """Return a :func:`filter_stderr` context manager for loaded asteroids.

    Builds filter prefixes from the 5 loaded asteroid designations so that
    kete's Rust-level ``"Impact detected between ..."`` warnings for
    self-colliding perturber asteroids are silently suppressed.

    Returns
    -------
    contextmanager
        A context manager that filters asteroid-impact stderr messages.
    """
    cache = get_kete_loaded_objects()
    filter_startswith = []
    for astnum_str in cache['asteroids']:
        filter_startswith.append(f'Impact detected between (Name("{astnum_str}"))  ')
    return filter_stderr(filter_startswith)
