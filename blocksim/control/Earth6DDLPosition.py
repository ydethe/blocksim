from datetime import datetime
from typing import Iterable

import numpy as np
from ahrs.utils import WMM

from ..utils import (
    FloatArr,
    build_local_matrix,
    quat_to_euler,
    quat_to_matrix,
    itrf_to_geodetic,
    deg,
)


class Earth6DDLPosition(object):
    """

    position: ITRF position (m)
    velocity: ITRF velocity (m/s)
    attitude: Attitude quaternion
    angular_rate: Angular rate

    """

    __slots__ = ["name", "time", "position", "velocity", "attitude", "angular_rate"]

    @classmethod
    def from_geodetic(
        cls,
        name: str = "",
        time: datetime = None,
        position: FloatArr = np.zeros(3),
        velocity: FloatArr = np.zeros(3),
        attitude: FloatArr = np.zeros(4),
        angular_rate: FloatArr = np.zeros(3),
    ) -> None:
        pass

    def __init__(
        self,
        name: str = "",
        time: datetime = None,
        position: FloatArr = np.zeros(3),
        velocity: FloatArr = np.zeros(3),
        attitude: FloatArr = np.zeros(4),
        angular_rate: FloatArr = np.zeros(3),
    ) -> None:
        self.name = name
        self.time = time
        self.position = position
        self.velocity = velocity
        self.attitude = attitude
        self.angular_rate = angular_rate

    def attitudeToQuaternion(self) -> FloatArr:
        return self.attitude

    def attitudeToDCM(self) -> FloatArr:
        return quat_to_matrix(*self.attitude)

    def attitudeToEuler(self) -> Iterable[float]:
        return quat_to_euler(*self.attitude)

    def positionToGeodetic(self) -> Iterable[float]:
        if self.position @ self.position < 1:
            return np.zeros(3)
        else:
            return itrf_to_geodetic(self.position)

    def positionToENU(self, origin: "Earth6DDLPosition") -> Iterable[float]:
        M = build_local_matrix(origin.position)
        return M.T @ (self.position - origin.position)

    def magneticDeclination(self, frame="ITRF") -> FloatArr:
        """

        Examples:
        >>> e = Earth6DDLPosition(time=datetime(2020,1,1),position=np.array([6378137.0, 0.0, 0.0]))
        >>> md = e.magneticDeclination()
        >>> (md*1e9).astype(np.int64)
        array([16022, -2248, 27536])

        """
        lon, lat, alt = self.positionToGeodetic()

        if np.abs(lat) < 1e-8:
            lat = 1e-8
        if np.abs(lon) < 1e-8:
            lon = 1e-8
        wm = WMM(date=self.time, latitude=deg(lat), longitude=deg(lon), height=alt)

        mn = wm.X * 1e-9
        me = wm.Y * 1e-9
        md = wm.Z * 1e-9

        if frame == "ENU":
            vm = np.array([me, mn, -md])
        elif frame == "NED":
            vm = np.array([mn, me, md])
        elif frame == "ITRF":
            M = build_local_matrix(self.position)
            vm = np.array([me, mn, -md])
            vm = M @ vm

        return vm

    def gravity(self, frame="ITRF") -> FloatArr:
        return np.zeros(3)


if __name__ == "__main__":
    from ..utils import rad, geodetic_to_itrf

    pos = geodetic_to_itrf(lon=rad(0), lat=rad(0), h=0)
    ep = Earth6DDLPosition(name="IMU", position=pos)
    print(ep.magneticDeclination())
