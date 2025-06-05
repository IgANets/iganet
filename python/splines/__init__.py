from pyiganet import pyiganet_core as _pyiganet_core


def UniformBSpline(geo, degrees, ncoeffs, init=None, core=None):
    """
    Creator function for uniform splines. Based on the input, a corresponding
    backend class will be created

    Parameters
    ----------
    geo: int
      geometric dimension
    degrees: list
      degrees in list of int.
    ncoeffs: list
      number of coefficients per dimension. list of int
    init: init

    Returns
    -------
    uniform_bspline: UniformBSpline
    """
    geo_str = f"{geo}d"
    deg_str = ""
    for d in degrees:
        deg_str += f"_{d}"

    if init is None:
        init = _pyiganet_core.splines.init.linear
    elif isinstance(init, int):
        init = _pyiganet_core.splines.init(init)

    if core is None:
        core = _pyiganet_core.core.Core

    return eval(f"_pyiganet_core.splines.UniformBSpline{geo_str}{deg_str}")(
        ncoeffs, init
    )
