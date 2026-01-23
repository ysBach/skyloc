import ctypes

import numpy as np
import spiceypy as sp

from .typeutil import empty_double_vector, str2char_p


__all__ = ["spkgps", "spkcvo"]


def spkgps(ref: str, obs: int, dummy_lt: bool = True):
    """Return boosted spkgps function.

    Parameters
    ----------
    ref : str
        Reference frame.
    obs : int
        Observer SPKID.
    dummy_lt : bool
        If `True`, light time will be set as dummy pointer and not returned.

    Returns
    -------
    spkgps_boosted : function
        Boosted spkgps function. Input arguments are `targ` and `et` ::
        - `targ` :  must be prepared by ``ctypes.c_int(int(spkid))``
        - `et` : must be prepared by ``ctypes.c_double(et)`` (use
            `timeutil.times2et` with ``return_c=True``).
    """
    ref = str2char_p(ref)
    obs = ctypes.c_int(obs)

    if dummy_lt:
        _lt = ctypes.c_double()

        def spkgps_boosted(targ, et):
            _ptarg = empty_double_vector(3)
            sp.libspice.spkgps_c(targ, et, ref, obs, _ptarg, ctypes.byref(_lt))
            return np.frombuffer(_ptarg, dtype=np.float64).copy()

    else:

        def spkgps_boosted(targ, et):
            _lt = ctypes.c_double()
            _ptarg = empty_double_vector(3)
            sp.libspice.spkgps_c(targ, et, ref, obs, _ptarg, ctypes.byref(_lt))
            return np.frombuffer(_ptarg, dtype=np.float64).copy(), _lt.value

    return spkgps_boosted


def spkcvo(
    outref: str,
    refloc: str,
    abcorr: str,
    obsctr: str,
    obsref: str,
    dummy_lt: bool = True,
):
    """Return boosted spkcvo function with obsepc = et.

    Parameters
    ----------
    outref : str
        Output reference frame, e.g., ``"ECLIPJ2000"``.
    refloc : str
        Reference location, e.g., ``"OBSERVER"``.
    abcorr : str
        Aberration correction.
    obsctr : str
        Observer center.
    obsref : str
        Observer reference frame.
    dummy_lt : bool
        If `True`, light time will be set as dummy pointer and not returned.

    Returns
    -------
    spkcvo_boosted : function
        Boosted spkcvo function. Input arguments are `target`, `obssta`, and
        `et`::
        - `target` must be prepared by ``str2char_p(str(spkid))``
        - `obssta` must be prepared by ``sp.stypes.to_double_vector(state)``
        - `et` must be prepared by `ctypes.c_double(et)` (use
            `timeutil.times2et` with ``return_c=True``).
    """
    outref = str2char_p(outref)
    refloc = str2char_p(refloc)
    abcorr = str2char_p(abcorr)
    obsctr = str2char_p(obsctr)
    obsref = str2char_p(obsref)

    if dummy_lt:
        _lt = ctypes.byref(ctypes.c_double())

        def spkcvo_boosted(target, obssta, et):
            state = empty_double_vector(6)
            # TODO: Can't I move `state`` outside the function safely...?
            # Reinitialize `state` every time the function is called takes 0.3us,
            # so 1.4M objects for 1k calls results in several minutes.
            sp.libspice.spkcvo_c(
                target,
                et,
                outref,
                refloc,
                abcorr,
                obssta,
                et,
                obsctr,
                obsref,
                state,
                _lt,
            )
            return np.frombuffer(state).copy()

    else:

        def spkcvo_boosted(target, obssta, et):
            _lt = ctypes.c_double()
            state = empty_double_vector(6)
            sp.libspice.spkcvo_c(
                target,
                et,
                outref,
                refloc,
                abcorr,
                obssta,
                et,
                obsctr,
                obsref,
                state,
                ctypes.byref(_lt),
            )
            return np.frombuffer(state).copy(), _lt.value

    return spkcvo_boosted
