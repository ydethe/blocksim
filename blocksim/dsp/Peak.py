from dataclasses import dataclass


@dataclass(init=True, repr=True)
class Peak:
    coord: tuple
    value: float
