import kete

__all__ = ["parse_frame"]


def parse_frame(frame):
    """Parse a frame input into a `kete.Frame` object.

    Parameters
    ----------
    frame : `~kete.Frames` or str
        The frame to parse. If str, currently the followings are available:
        {'Ecliptic', 'Equatorial', 'FK4', 'Galactic'}.

    """
    if isinstance(frame, kete.Frames):
        return frame
    if isinstance(frame, str):
        return getattr(kete.Frames, frame)
    raise ValueError(f"Unknown frame type: {frame}")
