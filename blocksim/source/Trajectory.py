from typing import Tuple

import numpy as np

from ..Logger import Logger


class Trajectory(object):
    def __init__(self, name: str, x: np.array, y: np.array, z: np.array, color: tuple):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.color = color

    @classmethod
    def fromLogger(
        cls, log: Logger, name: str, npoint: int, params: Tuple[str], color: tuple
    ) -> "Trajectory":
        xname, yname, zname = params

        if npoint > 0:
            x = log.getValue(xname)[:npoint]
            y = log.getValue(yname)[:npoint]
            z = log.getValue(zname)[:npoint]
        else:
            x = log.getValue(xname)
            y = log.getValue(yname)
            z = log.getValue(zname)

        traj = Trajectory(name=name, x=x, y=y, z=z, color=color)

        return traj

    def __len__(self):
        return len(self.x)
