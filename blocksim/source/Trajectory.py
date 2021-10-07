from typing import Tuple

import numpy as np

from ..utils import itrf_to_geodetic
from ..Logger import Logger


class Trajectory(object):
    """Trajectory with a name and a color"""

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

    def iterPosition(self):
        """Iterator through the positions.
        Each element of the iterator is a numpy array with elements (x,y,z)

        """
        for x, y, z in zip(self.x, self.y, self.z):
            yield np.array([x, y, z])

    def getGroundTrack(self) -> Tuple[np.array, np.array]:
        """Returns latitude and longitude array

        Returns:
          Longitudes array in rad
          Latitudes array in rad

        """
        ns = len(self)
        lat = np.empty(ns)
        lon = np.empty(ns)
        for k, pos in enumerate(self.iterPosition()):
            lat[k], lon[k], alt[k] = itrf_to_geodetic(pos)

        return lon, lat
