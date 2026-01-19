import ctypes

import numpy as np
import spiceypy as sp
from astropy.time import Time


def times2et(times, return_c=False, **kwargs):
    """Convert time to ET (in SPICE format).

    Parameters
    ----------
    times : str, list
        Time values that will be passed to ``~astropy.time.Time`` function.

    return_c : bool, optional
        If `True`, return an additional ET values as a list of ctypes.c_double.

    **kwargs : dict
        Additional arguments for ``~astropy.time.Time`` function.

    Returns
    -------
    times : astropy.time.Time
        Time object.

    ets : list
        List of ET (which means TDB in SPICE) values.

    ets_c : list
        List of ET values as ``ctypes.c_double``.
        Returned only if `return_c` is `True`.
    """
    times = Time(np.atleast_1d(times), **kwargs)
    # Vectorize using list comprehension (faster than append in loop)
    ets = [sp.str2et(_t) for _t in times.iso]
    if return_c:
        ets_c = [ctypes.c_double(_et) for _et in ets]
        return times, ets, ets_c
    return times, ets
