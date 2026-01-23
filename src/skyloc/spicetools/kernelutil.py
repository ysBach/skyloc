from pathlib import Path
from ..configs import PKG_PATH

__all__ = ["make_meta"]


# Path to spicetools/kernels:
DEFAULT_KERNELS = str(PKG_PATH / "kernels")

# Basic template for the meta file:
META_TEMPLATE = r"""\begintext
This meta file contains the paths to needed SPICE kernels.

\begindata
PATH_VALUES     = (
    {}
)
PATH_SYMBOLS    = (
    {}
)

KERNELS_TO_LOAD = (
    {}
)
"""


def make_meta(
    *args,
    path_values="",
    path_symbols="",
    autoreplace=False,
    output="kernel_meta",
    load=False,
):
    """Create a kernel meta file from a list of kernel paths.

    Parameters
    ----------
    *args : str
        List of kernel paths. Some files that are avaliable within this package
        can be specified by using ``$KERNELS``. For example,

        ::

        make_meta(
            "$KERNELS/lsk/naif0012.tls",
            "$KERNELS/pck/gm_de440.tpc",
            "$KERNELS/pck/pck00011.tpc"
        )

    path_values, path_symbols : str, path-like, or list of those, optional
        PATH_VALUES and corresponding PATH_SYMBOLS to be included in the meta
        file. Can be provided as a single string or a list of strings. They
        must have the same length. Whatever is provided, the first entry will
        be the default kernel path of SPICEtools (``kernels/`` directory) and
        the symbol ``"KERNELS"`` (which should be used with dollar sign
        as``"$KERNELS"`` when the user uses it in `args`).

    autoreplace : bool, optional
        If `True`, replace occurrences of `path_values` in `args` with
        corresponding `path_symbols`. Default is `False`.

    output : str, path-like, optional
        Name of the output kernel meta file.
        Default is ``kernel_meta`` and this file without further extension will
        be created in the current working directory.

    load : bool, optional
        If `True`, load the kernels after creating the meta file and return the
        handle.
        Default is `False`.

    Returns
    -------
    None
    """
    pv = [str(DEFAULT_KERNELS)]
    if path_values:
        if isinstance(path_values, str):
            path_values = [path_values]
        path_values = [str(p) for p in path_values]

    pv.extend(path_values)

    ps = ["KERNELS"]
    if path_symbols:
        if isinstance(path_symbols, str):
            path_symbols = [path_symbols]
        path_symbols = [str(s) for s in path_symbols]
    ps.extend(path_symbols)

    if len(path_values) != len(path_symbols):
        raise ValueError("`path_values` and `path_symbols` must have the same length.")

    pv_str = ",\n    ".join(f"'{str(p)}'" for p in pv)
    ps_str = ",\n    ".join(f"'{str(s)}'" for s in ps)

    if autoreplace:
        _args = []
        for i, _arg in enumerate(args):
            for val, sym in zip(pv, ps):
                _args.append(_arg.replace(val, sym))
        args = _args

    kernstrs = make_kernels_to_load(*args)  # create the list of kernels to load
    kerns = ",\n    ".join(kernstrs)

    with open(output, "w") as f:
        f.write(META_TEMPLATE.format(pv_str, ps_str, kerns))

    if load:
        import spiceypy as sp

        return sp.furnsh(str(output))


def make_kernels_to_load(*args, linemax=79):
    """Create a list of kernels to load from the given arguments.

    Parameters
    ----------
    *args : str
        List of kernel paths. If a kernel path is longer than `linemax` characters,
        it will be split into multiple strings following the SPICE format.
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html#Additional%20Meta-kernel%20Specifications

    linemax : int, optional
        Maximum number of characters in a line. Default is 79 for safety.

    Returns
    -------
    list
        List of kernel paths.
    """
    kernstrs = []
    if not args:
        raise ValueError("No kernel paths provided.")
    for arg in args:
        arg = str(arg)
        if len(arg) < linemax:
            kernstrs.append(f"'{arg}'")
        else:
            arg_spl = arg.split("/")
            _part = arg_spl[0]
            for part in arg_spl[1:]:
                if len(_part) + len(part) + 1 <= (linemax - 1):
                    _part += "/" + part
                else:
                    kernstrs.append(f"'{_part}/+'")
                    _part = part
            kernstrs.append(f"'{_part}'")

    return kernstrs
