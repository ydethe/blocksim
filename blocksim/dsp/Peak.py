from typing import Callable
from dataclasses import dataclass


@dataclass(init=True)
class Peak:
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
