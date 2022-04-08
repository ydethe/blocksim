from typing import Tuple

import pandas as pd
import numpy as np
from numpy import pi

from ..Logger import Logger
from ..utils import itrf_to_geodetic


class Trajectory(object):
    """Trajectory with a name and a color

    Args:
        name: Name of the trajectory
        color: The color as a 4-elements tuple:
            *  r between 0 and 1
            *  g between 0 and 1
            *  b between 0 and 1
            *  alpha between 0 and 1 (use 1 for fully opaque)
        t: array of time stamps
        x: array of ITRF X position
        y: array of ITRF Y position
        z: array of ITRF Z position

    """

    def __init__(
        self,
        name: str,
        color: tuple,
        t: "array" = np.array([]),
        x: "array" = np.array([]),
        y: "array" = np.array([]),
        z: "array" = np.array([]),
    ):
        self.name = name
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.color = color

    def ITRFToDataFrame(self) -> "DataFrame":
        """Converts the Trajectory into a pandas.DataFrame

        Returns:
            A DataFrame containing the time stamp, x, y, and z data (m)

        """
        df = pd.DataFrame(
            {"t": self.t, "x_itrf": self.x, "y_itrf": self.y, "z_itrf": self.z}
        )
        return df

    def GeodesicToDataFrame(self) -> "DataFrame":
        """Converts the Trajectory into a pandas.DataFrame

        Returns:
            A DataFrame containing the time stamp, longitude, latitude and altitude data (deg and m)

        """
        ns = len(self)
        lat = np.empty(ns)
        lon = np.empty(ns)
        alt = np.empty(ns)
        for k in range(ns):
            lo, la, alt[k] = itrf_to_geodetic((self.x[k], self.y[k], self.z[k]))
            lon[k] = lo * 180 / pi
            lat[k] = la * 180 / pi
        df = pd.DataFrame(
            {"t": self.t, "longitude": lon, "latitude": lat, "altitude": alt}
        )
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
        """Instanciates a Trajectory from a Logger

        Args:
            log: The Logger that contains the information
            name: Name of the Trajectory
            npoint: Number of samples to read. The npoint first samples are read
            params: A tuple with the 3 names of values (X, Y, Z) to read in log. These values shall be ITRF meter coordinates
            color: The color as a 4-elements tuple:

            *  r between 0 and 1
            *  g between 0 and 1
            *  b between 0 and 1
            *  alpha between 0 and 1 (use 1 for fully opaque)
            raw_value: True to use log.getRawValue. Otherwise, log.getValue is used (much slower)

        Returns:
            The Trajectory instance

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

    def addPosition(self, t: float, x: float, y: float, z: float):
        """Extends x, y and z arrays with one position

        Args:
            t: timestamp (s)
            x: X coordinate in ITRF (m)
            y: Y coordinate in ITRF (m)
            z: Z coordinate in ITRF (m)

        """
        self.x = np.hstack((self.x, np.array([x])))
        self.y = np.hstack((self.y, np.array([y])))
        self.z = np.hstack((self.z, np.array([z])))

    def iterPosition(self):
        """Iterator through the positions.
        Each element of the iterator is a numpy array with elements (t,x,y,z)

        Yields:
            array: The next (t,x,y,z) array

        """
        for t,x, y, z in zip(self.t,self.x, self.y, self.z):
            yield np.array([t,x, y, z])

    def getGroundTrack(self) -> Tuple["array", "array"]:
        """Returns latitude and longitude array

        Returns:
            A tuple of longitude and latitude array in deg

        """
        df = self.GeodesicToDataFrame()

        return np.array(df["longitude"]), np.array(df["latitude"])
