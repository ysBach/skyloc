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
    if return_c:
        ets = []
        ets_c = []
        for _t in times.iso:
            _et = sp.str2et(_t)
            _etc = ctypes.c_double(_et)
            ets.append(_et)
            ets_c.append(_etc)
        return times, ets, ets_c
    return times, [sp.str2et(_t) for _t in times.iso]
