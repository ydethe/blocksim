from numpy import pi

__all__ = ["deg", "rad"]


def deg(x: float) -> float:
    """Converts from radians to degrees

    Args:
      x
        Angle in radians

    Returns:
      Angle in degrees

    """
    return x * 180 / pi


def rad(x: float) -> float:
    """Converts from degrees to radians

    Args:
      x
        Angle in degrees

    Returns:
      Angle in radians

    """
    return x * pi / 180
