from typing import Tuple

import pandas as pd
import numpy as np
from numpy import pi

from blocksim.Logger import Logger

from ..utils import itrf_to_geodetic


class Trajectory(object):
    """Trajectory with a name and a color"""

    def __init__(
        self,
        name: str,
        color: tuple,
        x: "array" = np.array([]),
        y: "array" = np.array([]),
        z: "array" = np.array([]),
    ):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.color = color

    def ITRFToDataFrame(self) -> "DataFrame":
        df = pd.DataFrame({"x_itrf": self.x, "y_itrf": self.y, "z_itrf": self.z})
        return df

    def GeodesicToDataFrame(self) -> "DataFrame":
        ns = len(self)
        lat = np.empty(ns)
        lon = np.empty(ns)
        alt = np.empty(ns)
        for k in range(ns):
            lo, la, alt[k] = itrf_to_geodetic((self.x[k], self.y[k], self.z[k]))
            lon[k] = lo * 180 / pi
            lat[k] = la * 180 / pi
        df = pd.DataFrame({"longitude": lon, "latitude": lat, "altitude": alt})
        return df

    def __repr__(self):
        df = self.GeodesicToDataFrame()

        r = "Traj('%s')\n" % self.name
        r += df.__repr__()

        return r

    @classmethod
    def fromLogger(
        cls,
        log: Logger,
        name: str,
        npoint: int,
        params: Tuple[str],
        color: tuple,
        raw_value: bool = True,
    ) -> "Trajectory":
        """Creates a Trajectory from a Logger

        Args:
            log
                The Logger that contains the information
            name
                Name of the Trajectory
            npoint
                Number of samples to read. The npoint first samples are read
            params
                A tuple with the 3 names of values (X, Y, Z) to read in log. These values shall be ITRF meter coordinates
            color
                r,g,b,a tuple for the color of the trajectory
            raw_value
                True to use log.getRawValue. Otherwise, log.getValue is used (much slower)

        """
        xname, yname, zname = params

        if raw_value:
            mth = log.getRawValue
        else:
            mth = log.getValue

        if npoint > 0:
            x = mth(xname)[:npoint]
            y = mth(yname)[:npoint]
            z = mth(zname)[:npoint]
        else:
            x = mth(xname)
            y = mth(yname)
            z = mth(zname)

        traj = Trajectory(name=name, x=x, y=y, z=z, color=color)

        return traj

    def __len__(self):
        return len(self.x)

    def addPosition(self, x: float, y: float, z: float):
        """Extends x,y and z arrays wit one position

        Args:
            x (m)
                X coordinate in ITRF
            y (m)
                Y coordinate in ITRF
            z (m)
                Z coordinate in ITRF

        """
        self.x = np.hstack((self.x, np.array([x])))
        self.y = np.hstack((self.y, np.array([y])))
        self.z = np.hstack((self.z, np.array([z])))

    def iterPosition(self):
        """Iterator through the positions.
        Each element of the iterator is a numpy array with elements (x,y,z)

        """
        for x, y, z in zip(self.x, self.y, self.z):
            yield np.array([x, y, z])

    def getGroundTrack(self) -> Tuple["array", "array"]:
        """Returns latitude and longitude array

        Returns:
          Longitudes array in deg
          Latitudes array in deg

        """
        df = self.GeodesicToDataFrame()

        return np.array(df["longitude"]), np.array(df["latitude"])
