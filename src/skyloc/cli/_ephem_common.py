"""Shared helpers for ephemeris-oriented CLI tools."""

import re


DEFAULT_DEPOCH = 10000


def parse_horizons_step(step):
    """Parse fixed-duration Horizons step strings into an astropy TimeDelta."""
    from astropy import units as u
    from astropy.time import TimeDelta

    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z]+)\s*", step
    )
    if match is None:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    unit_map = {
        "s": u.s,
        "sec": u.s,
        "secs": u.s,
        "second": u.s,
        "seconds": u.s,
        "m": u.min,
        "min": u.min,
        "mins": u.min,
        "minute": u.min,
        "minutes": u.min,
        "h": u.hour,
        "hr": u.hour,
        "hrs": u.hour,
        "hour": u.hour,
        "hours": u.hour,
        "d": u.day,
        "day": u.day,
        "days": u.day,
        "w": u.week,
        "week": u.week,
        "weeks": u.week,
    }
    if unit not in unit_map:
        return None
    if value <= 0:
        raise ValueError("step must be positive")
    return TimeDelta(value * unit_map[unit])


def epochs_from_range(start, stop, step, *, in_tdb=False, depoch=DEFAULT_DEPOCH):
    """Build Horizons epochs, falling back to a JD list for long ranges."""
    if parse_horizons_step(step) is None:
        return {"start": start, "stop": stop, "step": step}

    epoch_count, jd_values = epoch_jds_from_range(start, stop, step, in_tdb=in_tdb)
    if epoch_count <= depoch:
        return {"start": start, "stop": stop, "step": step}
    return jd_values


def epoch_jds_from_range(start, stop, step, *, in_tdb=False):
    """Build the full inclusive JD sequence for a fixed-step time range."""
    from astropy import units as u
    from astropy.time import Time
    from erfa import ErfaWarning
    import numpy as np
    import warnings

    step_delta = parse_horizons_step(step)
    if step_delta is None:
        raise ValueError(
            "local ephemeris generation requires a fixed step such as "
            "'1d', '6h', '30m', or '10s'"
        )

    scale = "tdb" if in_tdb else "utc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        start_time = Time(start, scale=scale)
        stop_time = Time(stop, scale=scale)
        duration = stop_time - start_time
        interval_ratio = duration.to_value(u.s) / step_delta.to_value(u.s)
        n_intervals = int(np.floor(interval_ratio + 1.0e-12))

        if n_intervals < 0:
            raise ValueError("stop must be later than or equal to start")

        epoch_count = n_intervals + 1
        epoch_offsets = np.arange(epoch_count) * step_delta
        times = start_time + epoch_offsets

    return epoch_count, times.jd.tolist()


