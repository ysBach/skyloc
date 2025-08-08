from pathlib import Path

import kete
import numpy as np

from .statevec import make_kete_state

__all__ = [
    "FOVCollection",
    "make_fovlist",
    "make_rect_fov",
    "make_cone_fov",
    "make_omni_fov",
]


class FOVCollection:
    """A convenience tool for FOVList

    It allows (`fovc = FOVCollection(fovs)`):
        1. Easy indexing like numpy (e.g., `fovc[0]` or `fovc[1:10]`),
        2. Easy masking (e.g., `fovc[mask_arr]`),
        3. Easy access to FOV designations (e.g., `fovc["fov001"]`)
        4. Easy access by multiple designations (e.g., `fovc[["fov001", "fov002"]]`).

    Has useful methods/attributes:
      - `.fov_jds`: An array of JDs for the FOVs.
      - `.mask_by_desig(desigs)`: Returns a mask for the FOVs based on the
        designations. Very fast because it uses `set` behind the curtains.
          - Use the resulting mask as `fovc[mask]` to get the subset of FOVs in
            ndarray.
          - Use `fovc.fov_desigs[mask]` to get the corresponding designations
            in ndarray.
          - Make a new collection object by `FOVCollection(fovc[mask])`.
    """

    def __init__(self, fovs, fov_desigs=None, sort_fovlist=False):
        """
        Parameters
        ----------
        fovs : `kete.FOV` or iterable of `kete.FOV`
            An iterable of FOV objects, or a `kete.FOVList` object.

        fov_desigs : array-like of str, optional
            A list of designations for the FOVs. If not provided, the
            designations will be taken from the `observer.desig` of each FOV.
            Note that kete never allows non-str `desig` for observer `State`,
            so it is expected the given `fov_desigs` are strings.

        sort_fovlist : bool, optional
            If `True`, the FOVs will be sorted by JD (`kete.fov.FOVList.sort()`).
        """
        self.fovlist, self.fovarr = make_fovlist(fovs, sort=sort_fovlist)

        if fov_desigs is None:
            self.fov_desigs = []
            self.fov_jds = []
            self._desig_index = {}
            for i, fov in enumerate(self.fovlist):
                desig = fov.observer.desig
                self.fov_desigs.append(desig)
                self._desig_index[desig] = i
                self.fov_jds.append(fov.observer.jd)
            self.fov_desigs = np.asarray(self.fov_desigs)
            self.fov_jds = np.asarray(self.fov_jds)
        else:
            try:
                fov_desigs = list(fov_desigs)
            except Exception as e:
                raise ValueError(
                    "fov_desigs must be an iterable of strings, "
                    f"got {type(fov_desigs)}. \n{e}"
                )

            if len(fov_desigs) != len(self.fovlist):
                raise ValueError("Length of fov_desigs must match the length of fovs.")

            self.fov_desigs = np.asarray(fov_desigs)
            self.fov_jds = np.asarray([fov.observer.jd for fov in self.fovarr])
            self._desig_index = {str(desig): i for i, desig in enumerate(fov_desigs)}

    def __getitem__(self, item):
        """Get FOV by index or designation."""

        try:  # if int or numpy-like access:
            return self.fovarr[item]
        except IndexError:
            try:  # if str:
                return self.fovarr[self._desig_index[item]]
            except TypeError:  # list of str
                mask = self.mask_by_desig(item)
                return self.fovarr[mask]

    def __getattr__(self, name):
        # e.g., size
        try:
            return getattr(self.fovarr, name)
        except AttributeError:
            raise AttributeError(f"FOVCollection has no attribute {name}")

    def __repr__(self):
        return f"FOVCollection with {len(self.fovlist)} FOVs"

    def __len__(self):
        """Return the number of FOVs in the collection."""
        return len(self.fovlist)

    def mask_by_desig(self, desigs):
        """Return a mask for the FOVs based on the designations.

        Parameters
        ----------
        desigs : str or iterable of str
            A single designation or an iterable of designations to mask the FOVs.

        Returns
        -------
        fovmask : np.ndarray of bool
            A boolean mask indicating which FOVs have the given designations.

        Notes
        -----
        It will be faster than simple `FOVCollection.fov_desigs.isin(desigs)`
        or `[FOVCollection[desig] for desig in desigs]` because it uses
        `set` behind the curtains. For example, this method is about 10x faster
        than these alternatives for ~1000 FOV elements.
        """
        if isinstance(desigs, str):
            desigs = [desigs]

        # Below uses set behind the curtains: ~10x faster than np.isin for
        # ~1000 elements
        _set = set(desigs)
        fovmask = np.fromiter(
            (k in _set for k in self.fov_desigs), dtype=bool, count=len(self.fov_desigs)
        )

        return fovmask

    def save(self, filename):
        """Save the FOVCollection.

        Notes
        -----
        It only saves `self.fovlist`, by using `kete.fov.FOVList.save()`. Since
        `FOVCollection` can be initiated solely from `FOVList`, it is used to
        be loaded back.
        """
        self.fovlist.save(filename)

    @classmethod
    def load(cls, filename):
        """Load the FOVCollection from a file.

        Parameters
        ----------
        filename : str
            Path to the file containing the saved FOVList

        Returns
        -------
        FOVCollection
            A new FOVCollection instance loaded from the file

        Notes
        -----
        This is a class method that can be called as FOVCollection.load(filename).
        It loads the FOVList from the file and creates a new FOVCollection instance.
        """
        _fovl = kete.fov.FOVList.load(filename)
        return cls(_fovl)


def make_fovlist(fovs, output=None, overwrite=False, sort=False):
    """Convenience function to create a `kete.FOVList` object from an iterable of
    `kete.FOV` objects.

    Parameters
    ----------
    fovs : iterable of `kete.FOV`
        An iterable of FOV objects.

    output : path-like, optional
        If provided, the FOVList will be saved to this file.

    overwrite : bool, optional
        If `True`, overwrite the existing file if it exists.
        Default is `False`.

    sort : bool, optional
        If `True`, the FOVList will be sorted by JD (`kete.fov.FOVList.sort()`).
        Default is `False`.

    Returns
    -------
    FOVList : `kete.FOVList`
        A FOVList object containing the FOVs.

    fovs : np.ndarray of `kete.FOV`
        A list of FOV objects, `np.asarray(FOVList)` for convenience (e.g.,
        indexing, masking and slicing)

    Notes
    -----
    Preferred way: use `FOVCollection`.
    """
    if output is None:
        out_exists = False
    else:
        out_exists = Path(output).exists()
        output = str(output)

    if out_exists and not overwrite:
        # If the output file exists and overwrite is False, load it
        fovlist = kete.fov.FOVList.load_parquet(str(output))
        if sort:
            fovlist.sort()
        return fovlist, np.asarray(fovs)

    if isinstance(fovs, kete.fov.FOVList):
        fovlist = fovs
    elif isinstance(fovs, (kete.RectangleFOV, kete.ConeFOV, kete.OmniDirectionalFOV)):
        fovlist = kete.fov.FOVList((fovs,))
    else:
        try:
            fovlist = kete.fov.FOVList(tuple(fovs))
        except Exception:
            fovlist = kete.fov.FOVList(list(fovs))

    if sort:
        fovlist.sort()

    if out_exists or overwrite:
        fovlist.save_parquet(output)
        # Just to be sure, load it back for consistency
        fovlist = kete.fov.FOVList.load_parquet(output)

    return fovlist, np.asarray(fovlist)


def make_rect_fov(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    center_ra_deg,
    center_dec_deg,
    rotation_deg,
    lon_width_deg=0.1,
    lat_width_deg=0.1,
    pos_frame=kete.Frames.Ecliptic,
    vel_frame=kete.Frames.Ecliptic,
    center_id=399,
    fov_to_sun=True,
):
    """Make kete RectangleFOV

    Parameters
    ----------
    state_desig : str
        Name of the observer `State` used in the RectanlgeFOV object.

    jd_tdb : float or `kete.Time`
        The time of the state in TDB jd time, see `kete.Time`.

    pos_au, vel_aupd : array_like
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).

    center_ra_deg, center_dec_deg : float
        The center of the FOV in equatorial J2000 RA/DEC in degrees.

    rotation_deg : float
        The rotation of the field of view in degrees.

    lon_width_deg, lat_width_deg : float, optional
        The width of the FOV in degrees, by default 0.1 degrees in both
        longitude and latitude.

    pos_frame, vel_frame : `kete.Frames`, optional
        The frame of the position and velocity, by default
        `kete.Frames.Ecliptic`.

    center_id : int, optional
        The center of the FOV, by default 399 (Earth geocenter; NOT Earth-Moon
        barycenter (``3``)).

    fov_to_sun : bool, optional
        If True, the field of view is centered on the Sun, by default True.
        Note that kete's fov must be centered around the Sun to avoid any
        "surprises" when checking an object is in the FOV.


    Returns
    -------
    obssta : `kete.State`
        The state of the observer.

    fov : `kete.RectangleFOV`
        The field of view.
    """
    obssta = make_kete_state(
        state_desig=state_desig,
        jd_tdb=jd_tdb,
        pos_au=pos_au,
        vel_aupd=vel_aupd,
        pos_frame=pos_frame,
        vel_frame=vel_frame,
        center_id=center_id,
    )
    if fov_to_sun:
        obssta = obssta.change_center(10)

    fov = kete.RectangleFOV(
        pointing=kete.Vector.from_ra_dec(ra=center_ra_deg, dec=center_dec_deg),
        rotation=rotation_deg,
        observer=obssta,
        lon_width=lon_width_deg,
        lat_width=lat_width_deg,
    )
    return obssta, fov


def make_cone_fov(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    center_ra_deg,
    center_dec_deg,
    radius_deg=0.1,
    pos_frame=kete.Frames.Ecliptic,
    vel_frame=kete.Frames.Ecliptic,
    center_id=399,
    fov_to_sun=True,
):
    """Make kete ConeFOV

    Parameters
    ----------
    state_desig : str
        Name of the observer `State` used in the ConeFOV object.

    jd_tdb : float or `kete.Time`
        The time of the state in TDB jd time, see `kete.Time`.

    pos_au, vel_aupd : array_like
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).

    center_ra_deg, center_dec_deg : float
        The center of the FOV in equatorial J2000 RA/DEC in degrees.

    rotation_deg : float
        The rotation of the field of view in degrees.

    radius_deg : float, optional
        The radius of the FOV in degrees, by default 0.1 degrees.

    pos_frame, vel_frame : `kete.Frames`, optional
        The frame of the position and velocity, by default
        `kete.Frames.Ecliptic`.

    center_id : int, optional
        The center of the FOV, by default 399 (Earth geocenter; NOT Earth-Moon
        barycenter (``3``)).

    fov_to_sun : bool, optional
        If True, the field of view is centered on the Sun, by default True.
        Note that kete's fov must be centered around the Sun to avoid any
        "surprises" when checking an object is in the FOV.

    Returns
    -------
    obssta : `kete.State`
        The state of the observer.

    fov : `kete.ConeFOV`
        The field of view.
    """
    obssta = make_kete_state(
        state_desig=state_desig,
        jd_tdb=jd_tdb,
        pos_au=pos_au,
        vel_aupd=vel_aupd,
        pos_frame=pos_frame,
        vel_frame=vel_frame,
        center_id=center_id,
    )
    if fov_to_sun:
        obssta = obssta.change_center(10)

    fov = kete.ConeFOV(
        pointing=kete.Vector.from_ra_dec(ra=center_ra_deg, dec=center_dec_deg),
        angle=radius_deg,
        observer=obssta,
    )

    return obssta, fov


def make_omni_fov(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    pos_frame=kete.Frames.Ecliptic,
    vel_frame=kete.Frames.Ecliptic,
    center_id=399,
    fov_to_sun=True,
):
    """Make kete OmniDirectionalFOV

    Parameters
    ----------
    state_desig : str
        Name of the observer `State` used in the OmniDirectionalFOV object.

    jd_tdb : float or `kete.Time`
        The time of the state in TDB jd time, see `kete.Time`.

    pos_au, vel_aupd : array_like
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).

    pos_frame, vel_frame : `kete.Frames`, optional
        The frame of the position and velocity, by default
        `kete.Frames.Ecliptic`.

    center_id : int, optional
        The center of the FOV, by default 399 (Earth geocenter; NOT Earth-Moon
        barycenter (``3``)).

    fov_to_sun : bool, optional
        If True, the field of view is centered on the Sun, by default True.
        Note that kete's fov must be centered around the Sun to avoid any
        "surprises" when checking an object is in the FOV.

    Returns
    -------
    obssta : `kete.State`
        The state of the observer.

    fov : `kete.OmniDirectionalFOV`
        The field of view.
    """
    obssta = make_kete_state(
        state_desig=state_desig,
        jd_tdb=jd_tdb,
        pos_au=pos_au,
        vel_aupd=vel_aupd,
        pos_frame=pos_frame,
        vel_frame=vel_frame,
        center_id=center_id,
    )
    if fov_to_sun:
        obssta = obssta.change_center(10)

    fov = kete.OmniDirectionalFOV(observer=obssta)

    return obssta, fov
