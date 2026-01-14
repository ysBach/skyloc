from pathlib import Path
from urllib import request
import base64
import requests
from astropy.table import vstack
import numpy as np
import kete
from astropy import units as u
from astropy import constants as c
from astropy.table import Table

from .utils import tdb2utc
from .keteutils import parse_frame, is_spk_loaded
from .configs import HORIZONS_DEPOCHS, PKG_PATH


def iterator(it):
    try:
        from tqdm import tqdm

        return tqdm(it)
    except ImportError:
        return it


__all__ = [
    "download_jpl_de",
    "HorizonsSPKQuery",
    "horizons_vector",
    "horizonsvec2ketestate",
    "horizons_quick",
]


def download_jpl_de(dename="de440s", output=None, overwrite=False):
    """Download JPL development ephemeris file (intended to be used one time).

    Parameters
    ----------
    dename : str
        Name of the ephemeris file to download. Default is "de440s".
        See https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/ for available files.

    output : str, pathlib.Path, optional
        Name of the output file path. If not provided, it will be saved into
        this package's `kernels/` directory.

    overwrite : bool, optional
        If `True`, overwrite the existing file if it exists. Default is
        `False`, i.e., not download any file but returns the path to the
        existing one.

    Returns
    -------
    output : pathlib.Path
        Path to the downloaded or existing ephemeris file.

    existed : bool
        `True` if the file already existed, `False` if it was downloaded.
    """
    if not dename.endswith(".bsp"):
        dename += ".bsp"

    if output is None:
        # output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", dename)
        output = PKG_PATH / "kernels" / dename
    else:
        output = Path(output)

    if output.exists() and not overwrite:
        return output, True

    # Ensure the output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Download the file
    url = f"https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/{dename}"
    response = request.urlopen(url)

    with open(output, "wb") as f:
        f.write(response.read())

    # print(f"Downloaded {dename} to {output}")
    return output, False


class HorizonsSPKQuery:
    """Class to handle JPL Horizons SPK queries."""

    def __init__(self, command, start=None, stop=None, obj_data=False, output=None):
        """Get SPK query parameters for JPL Horizons.

        Parameters
        ----------
        command : str
            The ``COMMAND`` parameter in Horizons Query API. See the
            ``COMMAND`` Parameter section in this link for the details
            https://ssd-api.jpl.nasa.gov/doc/horizons.html#ephem_type
            For small bodies, it should generally be ending with a semicolon
            (``";"``), and in the format of one of these::
            - ``"<astnum>;"``, e.g., ``"99942;"``
            - ``"<name>;"`` (e.g., ``"Apophis;"``),
            - ``"DES=<des>;"`` (e.g., ``"DES=1999 AN10;"``).
            - ``"DES=<spkid>;"`` (e.g., ``"DES=20099942;"``).

        start, stop : str, optional
            Start and stop times of the query in ISO format. If not provided, the
            current time and one day later will be used (the default setting of
            Horizons API).

        obj_data : bool, optional
            If `True`, include object data in the SPK file.

        """
        self.base_url = (
            "https://ssd.jpl.nasa.gov/api/horizons.api?format=json&EPHEM_TYPE=SPK"
        )
        if not isinstance(command, str):
            raise TypeError("`command` must be str")

        self._params = {
            "COMMAND": f"'{command}'",
            "START_TIME": start,
            "STOP_TIME": stop,
            "OBJ_DATA": "YES" if obj_data else "NO",
        }

        self.output = output

    def query(self, decode=True):
        response = requests.get(self.base_url, params=self._params)
        self.url = response.url
        if not response.ok:
            raise ValueError(f"Query failed: {response.text}")

        data = response.json()
        if data["signature"]["version"] != "1.2":
            raise ValueError(
                f"Only ver 1.2 is supported but got {data['signature']['version']=}"
            )

        # If the request was valid...
        try:
            self.spk = data["spk"]
            if not self.spk.startswith("REFGL1NQ"):
                raise ValueError(
                    "Invalid SPK data: It does not start with REFGL1NQ (DAF/SPK)."
                )
            if decode:
                self.spk = base64.b64decode(self.spk)
        except KeyError:
            raise ValueError(f"The key 'spk' is not found in the response: {data}")

        if self.output is not None:
            with open(self.output, "wb" if decode else "w") as f:
                f.write(self.spk)
            # Logger.log(f"SPK data written to {self.output}")


def horizons_vector(
    epochs,
    obsid,
    location=None,
    id_type=None,
    depochs=HORIZONS_DEPOCHS,
    aberrations="geometric",
    refplane="ecliptic",
    spice_units=False,
    invert=False,
    return_arr=False,
    try_spice=False,
    **kwargs,
):
    """Get the state vector from JPL Horizons ("vector query").

    Parameters
    ----------
    epochs : scalar, list-like, or dictionary
        Either a list of epochs in JD or MJD format or a dictionary defining a
        range of times and dates; the range dictionary has to be of the form
        {``'start'``: 'YYYY-MM-DD [HH:MM:SS]', ``'stop'``: 'YYYY-MM-DD
        [HH:MM:SS]', ``'step'``: 'n[y|d|m|s]'}. Epoch timescales should be TDB
        for vector queries. If `None` (default), the current time is used.
        If `obsid` and `location` are available as loaded SPK **and**
        `try_spice` is `True`, `epochs` must be in JD (units of days).

    obsid : str or dict
        Name, number, or designation of target object. Uses the same codes as
        JPL Horizons. Arbitrary topocentric coordinates can be added in a dict.
        The dict has to be of the form {``'lon'``: longitude in deg (East
        positive, West negative), ``'lat'``: latitude in deg (North positive,
        South negative), ``'elevation'``: elevation in km above the reference
        ellipsoid, [``'body'``: Horizons body ID of the central body; optional;
        if this value is not provided it is assumed that this location is on
        Earth]}.  Float values are assumed to have units of degrees and
        kilometers.

    location : str or dict, optional
        Center body name (for vector queries). Uses the same codes as JPL Horizons.
        If `None` (default), the Sun's center is used. Arbitrary topocentric
        coordinates for ephemerides queries can be provided in the format of a
        dictionary. The dictionary has to be of the form {``'lon'``: longitude
        (East positive, West negative), ``'lat'``: latitude (North positive,
        South negative), ``'elevation'``: elevation above the reference
        ellipsoid, [``'body'``: Horizons body ID of the central body; optional;
        if this value is not provided it is assumed that this location is on
        Earth]}. Float values are assumed to have units of degrees and
        kilometers.
        Some tips: Sun center is ``"@10"``.

    id_type : str, optional
        Controls Horizons's object selection for ``id``
        [HORIZONSDOC_SELECTION]_ . Options: ``'designation'`` (small body
        designation), ``'name'`` (asteroid or comet name), ``'asteroid_name'``,
        ``'comet_name'``, ``'smallbody'`` (asteroid and comet search), or
        `None``(first search search planets, natural satellites, spacecraft,
        and special cases, and if no matches, then search small bodies).

    depochs : int, optional
        Number of epochs in a chunk to use for the vector query. This is
        necessary to avoid "too long URI" errors when `epochs` is a large list.
        Default is `HORIZONS_DEPOCHS`.

    aberrations : {'geometric', 'astrometric', 'apparent'}, optional
        Aberrations to be accounted for.
        Ignored if `obsid` and `location` are available as loaded SPK **and**
        `try_spice` is `True` (result must be corresponding to `'geometric'`).
        Default : 'geometric'.

    refplane : {'ecliptic', 'earth', 'body'}, optional
        Reference plane for all output quantities: ``'ecliptic'`` (ecliptic and
        mean equinox of reference epoch), ``'earth'`` (Earth mean equator and
        equinox of reference epoch), or ``'body'`` (body mean equator and node
        of date).
        Will be converted as {'ecliptic': 'Ecliptic', 'earth': 'Equatorial'}
        when `obsid` and `location` are available as loaded SPK **and**
        `try_spice` is `True`.

        Default: ``'ecliptic'``.

        See "Horizons Reference Frames" in the astroquery documentation for
        details.

    spice_units : bool, optional
        If `True`, the output table will be in SPICE units (km for distance and
        km/s for velocity.). Otherwise, use default JPL Horizons units (au for
        distance and au/day for velocity, as of writing).
        Default: `False`

    invert : bool, optional
        If `True`, the position and velocity vectors will be inverted.
        Useful in this case: The user only has the TDB time at the observatory
        in space far from the NAIF object (e.g., Earth). Then the easiest way
        to query the state vector of the observatory is to query the vector
        **to** the geocenter and invert it.

    return_arr : bool, optional
        If `True`, return the state vector components as two numpy arrays
        (position and velocity), on top of the full table.

    try_spice : bool, optional
        If `True`, attempt to use SPICE kernels to avoid query by parsing
        `obsid` and `location`. This can provide faster results for major
        bodies with loaded SPK (see `skyloc.KETE_LOADED_SPKS` and
        `is_spk_loaded`).

    **kwargs : dict, optional
        Additional keyword arguments to pass to
        `~astroquery.jplhorizons.Horizons.vectors`. See
        https://astroquery.readthedocs.io/en/latest/api/astroquery.jplhorizons.HorizonsClass.html#astroquery.jplhorizons.HorizonsClass.vectors

    Returns
    -------
    astropy.Table
        The JPL Horizons vector query result.

    pos : numpy.ndarray, optional
        The position vector in au or km, if `return_arr` is `True`.

    vel : numpy.ndarray, optional
        The velocity vector in au/day or km/s, if `return_arr` is `True`.
    """
    from astroquery.jplhorizons import Horizons

    is_loaded_obs, _obsid = is_spk_loaded(obsid)
    is_loaded_loc, _locid = is_spk_loaded(location)
    if try_spice and is_loaded_obs and is_loaded_loc:
        # Do not query at all
        _frame = parse_frame({"ecliptic": "Ecliptic", "earth": "Equatorial"}[refplane])
        pos = []
        vel = []
        vecs = {c: [] for c in ["x", "y", "z", "range", "vx", "vy", "vz", "range_rate"]}
        for _jd in epochs:
            _sta = kete.spice.get_state(_obsid, _jd, _locid, _frame)
            pos.append(_sta.pos)
            vel.append(_sta.vel)

        pos = np.array(pos) << u.au
        vel = np.array(vel) << u.au / u.day

        # Mimic Horizons query results (NOTE: Do we really have to....?)
        vecs = Table()
        vecs["datetime_jd"] = epochs << u.d
        vecs["targetname"] = obsid
        vecs["x"] = pos[:, 0]
        vecs["y"] = pos[:, 1]
        vecs["z"] = pos[:, 2]
        vecs["range"] = np.linalg.norm(pos, axis=1)
        vecs["vx"] = vel[:, 0]
        vecs["vy"] = vel[:, 1]
        vecs["vz"] = vel[:, 2]
        _dot_r_v = np.einsum("ij,ij->i", pos, vel)
        vecs["range_rate"] = _dot_r_v / vecs["range"]
        vecs["lighttime"] = (vecs["range"] / (c.c)).to(u.day)  # sec

        vecs["tdb_in"] = epochs  # input times

    else:
        horkw = dict(id=obsid, location=location, id_type=id_type)
        # if epochs is iterable, split it into chunks of size `depochs`.
        if isinstance(epochs, (str, dict)):
            obj = Horizons(epochs=epochs, **horkw)
            vecs = obj.vectors(aberrations=aberrations, refplane=refplane, **kwargs)
        elif hasattr(epochs, "__iter__"):
            vecs = []
            for i in range(0, len(epochs), depochs):
                _epochs = epochs[i : i + depochs]
                obj = Horizons(epochs=_epochs, **horkw)
                vec = obj.vectors(aberrations=aberrations, refplane=refplane, **kwargs)
                vec["tdb_in"] = _epochs
                # ^^^^^^^^^^^ add exact input values ("datetime_jd" may slightly differ)
                vecs.append(vec)
            vecs = vstack(vecs)
        else:
            raise TypeError(
                f"`epochs` must be str, dict, or iterable; got {type(epochs)}"
            )

    if spice_units:
        # Convert to SPICE units (km and km/s)
        try:
            vecs["lighttime"] = vecs["lighttime"].to(u.s)
            for col in ("x", "y", "z", "range"):
                vecs[col] = vecs[col].to(u.km)
            for col in ("vx", "vy", "vz", "range_rate"):
                vecs[col] = vecs[col].to(u.km / u.s)
        except (ValueError, u.UnitConversionError) as e:
            raise ValueError(
                "Failed to convert units to SPICE (km and km/s). "
                "Check with `spice_units=False` to ensure the input is "
                "in the correct format, or if JPL output is correct."
            ) from e

    if invert:
        # Invert position and velocity vectors
        for xyz in ("x", "y", "z"):
            vecs[xyz] = -vecs[xyz]
            vecs[f"v{xyz}"] = -vecs[f"v{xyz}"]

    if return_arr:
        pos = np.array([vecs["x"].value, vecs["y"].value, vecs["z"].value]).T
        vel = np.array([vecs["vx"].value, vecs["vy"].value, vecs["vz"].value]).T
        return vecs, pos, vel

    return vecs


def horizonsvec2ketestate(
    vec, desigs=None, frame=kete.Frames.Ecliptic, center_id=10, invert=False
):
    """Convert JPL Horizons vector query result to `kete` state vector.

    Parameters
    ----------
    vec : astropy.Table
        The JPL Horizons vector query result.

    desigs : list, optional
        A list of designations for the objects. If not provided, the
        ``targetname`` column in the vector query result will be used.
        Default is `None`.

    frame : `~kete.Frames`, str, optional
        The coordinate frame to use for the state vector. Default is
        `kete.Frames.Ecliptic`.

    center_id : int, optional
        The SPICE kernel ID which defines the central reference point, defaults
        to the Sun (10).

    invert : bool, optional
        If `True`, the position and velocity vectors will be inverted.
        Useful in this case: The user only has the TDB time at the observatory
        in space far from the NAIF object (e.g., Earth). Then the easiest way
        to query the state vector of the observatory is to query the vector
        **to** the geocenter and invert it.

    Returns
    -------
    states : `~kete.SimultaneousStates`
        The `kete` state vector object.
    """
    if desigs is None:
        desigs = vec["targetname"]
    # Convert to Kete State objects
    pos = np.stack([vec[c].to("au").value for c in ("x", "y", "z")], axis=-1)
    vel = np.stack([vec[c].to("au/day").value for c in ("vx", "vy", "vz")], axis=-1)

    if invert:
        pos = -pos
        vel = -vel

    return [
        kete.State(
            desig=d,
            jd=t,
            pos=kete.Vector(p, frame=parse_frame(frame)),
            vel=kete.Vector(v, frame=parse_frame(frame)),
            center_id=center_id,
        )
        for d, p, v, t in zip(desigs, pos, vel, vec["tdb_in"])
    ]


def horizons_quick(
    objid, epochs, depochs=HORIZONS_DEPOCHS, location="500", in_tdb=True, **kwargs
):
    """Quick query for the object ID from JPL Horizons to compare with kete.

    Parameters
    ----------
    objid : str
        The object ID to be queried.

    epochs : list-like
        The epochs to be queried.

    depochs : int, optional
        Number of epochs in a chunk to use for the query. This is necessary
        to avoid "too long URI" errors when `epochs` is a large list.
        Default is `HORIZONS_DEPOCHS`.

    location : str, optional
        The location of the observer. Default is "500".

    in_tdb : bool, optional
        If `True`, the input epochs are in TDB. Default is `True`.
        If `False`, the input epochs are in UTC.

    **kwargs : dict, optional
        Additional keyword arguments to pass to
        `~astroquery.jplhorizons.Horizons.ephemerides`. See
        https://astroquery.readthedocs.io/en/latest/api/astroquery.jplhorizons.HorizonsClass.html#astroquery.jplhorizons.HorizonsClass.ephemerides
    """
    from astroquery.jplhorizons import Horizons

    if in_tdb:
        epochs = tdb2utc(epochs).jd

    eph = []

    for i in range(0, len(epochs), depochs):
        _epochs = epochs[i : i + depochs]
        obj = Horizons(id=objid, location=location, epochs=_epochs)
        _eph = obj.ephemerides(
            extra_precision=True, quantities=",".join(map(str, range(1, 49))), **kwargs
        )
        eph.append(_eph)
    eph = vstack(eph)
    colmaps = {
        "datetime_jd": "jd_utc",
        "RA": "ra",
        "DEC": "dec",
        "alpha_true": "alpha",
        "RA_rate": "dra*cosdec/dt",
        "DEC_rate": "ddec/dt",
        "r": "r_hel",
        "delta": "r_obs",
        "EclLon": "hel_ecl_lon",
        "EclLat": "hel_ecl_lat",
        "ObsEclLon": "obs_ecl_lon",
        "ObsEclLat": "obs_ecl_lat",
        "V": "vmag",
        "Tmag": "Tmag",
        "Nmag": "Nmag",
        "Sky_motion": "sky_motion",
        "Sky_mot_PA": "sky_motion_pa",
    }
    eph2compare = eph.to_pandas()
    eph2compare = eph2compare.loc[
        :, eph2compare.columns.isin(["datetime_jd"] + list(colmaps.keys()))
    ]
    eph2compare = eph2compare.rename(columns=colmaps)
    eph2compare["dra*cosdec/dt"] /= 60  # arcsec/h to arcsec/min
    eph2compare["ddec/dt"] /= 60  # arcsec/h to arcsec/min
    return eph, eph2compare
