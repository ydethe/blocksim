import numpy as np

from .exceptions import *
from . import logger


__all__ = ["deg", "rad"]


def deg(x: float) -> float:
    """Converts from radians to degrees

    Args:
      x
        Angle in radians

    Returns:
      Angle in degrees

    """
    return x * 180 / np.pi


def rad(x: float) -> float:
    """Converts from degrees to radians

    Args:
      x
        Angle in degrees

    Returns:
      Angle in radians

    """
    return x * np.pi / 180


def assignVector(
    v: np.array, expected_shape: tuple, dst_name: str, src_name: str, dtype
) -> np.array:
    """

    Args:
      v
        np.array to assign
      expected_shape
        Expected shape for v
      dst_name
        Name of the element where the assignement will take place. (To allow meaningfull error messages)
      src_name
        Name of the source vector. (To allow meaningfull error messages)
      dtype
        Type of the assigned vector

    Returns:
      Copy of the vector v if no problem encountered

    Raises:
      ValueError
        If the vector is not a np.array or not with the correct shape

    Examples:
      >>> v = np.arange(5)
      >>> assignVector(v, (5,), 'elem', 'v', np.float64)
      array([0., 1., 2., 3., 4.])

    """
    if isinstance(v.shape, int):
        vshape = (v.shape,)
    else:
        vshape = v.shape

    if isinstance(expected_shape, int):
        expected_shape = (expected_shape,)

    if type(v) != type(np.empty(1)):
        txt = "Element '%s' : Argument '%s'=%s is not a vector" % (
            dst_name,
            src_name,
            v,
        )
        logger.error(txt)
        raise InvalidAssignedVector(txt)

    elif vshape != expected_shape:
        txt = "Element '%s' : Array '%s'=%s has shape %s; expected %s" % (
            dst_name,
            src_name,
            v,
            vshape,
            expected_shape,
        )
        logger.error(txt)
        raise InvalidAssignedVector(txt)

    else:
        return np.array(v.copy(), dtype=dtype)
