from typing import Callable
from dataclasses import dataclass


@dataclass(init=True)
class Peak:
    """Represents a peak in a plot or a 3D plot

    Examples:
        >>> p = Peak(coord_label=("x",), coord_unit=("m",),coord=(1,), value=3.5)
        >>> print(p)
        Peak(x=1 m, value=3.5)

    """

    coord_label: tuple
    coord_unit: tuple
    coord: tuple
    value: float

    def __repr__(self):
        from ..graphics import format_parameter

        res = "%s(" % self.__class__.__name__
        for lbl, c, unt in zip(self.coord_label, self.coord, self.coord_unit):
            txt = format_parameter(c, unt)
            res += "%s=%s, " % (lbl, txt)
        res += "value=%.3g)" % self.value

        return res
