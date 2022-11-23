from abc import abstractmethod
from typing import Tuple, List, Union, Any
from pathlib import Path
from datetime import datetime, timedelta, timezone
import requests

from nptyping import NDArray, Shape
import numpy as np
from numpy import cos, sin, tan, sqrt
import scipy.linalg as lin
from sgp4.api import Satrec, WGS84
from sgp4.functions import days2mdhms

from ..Simulation import Simulation
from ..control.Route import Group
from ..core.Node import AComputer
from .. import logger
from ..constants import *
from ..utils import (
    orbital_to_teme,
    itrf_to_geodetic,
    teme_to_orbital,
    orbital_to_teme,
    teme_to_itrf,
    itrf_to_teme,
    time_to_jd_fraction,
)
from .Trajectory import Trajectory


__all__ = ["ASatellite", "CircleSatellite", "SGP4Satellite", "createSatellites"]


def sgp4_to_teme(satrec: Satrec, t_epoch: float) -> NDArray[Any, Any]:
    """
    TEME : https://en.wikipedia.org/wiki/Earth-centered_inertial#TEME

    Args:
        satrec: a Satrec instance from sgp4.api
        t_epoch: Time since 31/12/1949 00:00 UT (s)

    Returns:
        A 6-elements array with 3 position scalars (m) and 3 velocity scalar (m/s) in TEME frame

    """
    jd, fraction = time_to_jd_fraction(t_epoch)

    e, rTEME, vTEME = satrec.sgp4(jd, fraction)
    if e == 1:
        raise AssertionError("Mean eccentricity is outside the range 0 ≤ e < 1.")
    elif e == 2:
        raise AssertionError("Mean motion has fallen below zero.")
    elif e == 3:
        raise AssertionError("Perturbed eccentricity is outside the range 0 ≤ e ≤ 1.")
    elif e == 4:
        raise AssertionError("Length of the orbit’s semi-latus rectum has fallen below zero.")
    elif e == 6:
        raise AssertionError(
            "Orbit has decayed: the computed position is underground. (The position is still returned, in case the vector is helpful to software that might be searching for the moment of re-entry.)"
        )

    rTEME = np.array(rTEME)  # km
    vTEME = np.array(vTEME)  # km/s

    pv = np.empty(6)
    pv[:3] = rTEME * 1000
    pv[3:] = vTEME * 1000
    return pv


class ASatellite(AComputer):
    """
    Abstract class to describe a satellite

    The outputs of the computer are **itrf** and **subpoint**

    itrf:

    * px: X coordinate in geocentric ITRF (m)
    * py: Y coordinate in geocentric ITRF (m)
    * pz: Z coordinate in geocentric ITRF (m)
    * vx: X coordinate of velocity in geocentric ITRF (m/s)
    * vy: Y coordinate of velocity in geocentric ITRF (m/s)
    * vz: Z coordinate of velocity in geocentric ITRF (m/s)

    subpoint:

    * lon: Longitude (rad)
    * lat: Latitude (rad)

    Attributes:
        tsync: Datetime object that gives the date and time at simulation time 0
        orbit_mano: mean anomaly at tsync (rad)
        orbit_eccentricity: eccentricty
        orbit_semi_major_axis: semi major axis (m)
        orbit_inclination: Inclination(rad)
        orbit_argp: Argument of perigee (rad)
        orbit_node: R.A. of ascending node (rad)
        orbit_bstar: drag coefficient (1/earth radii)
        orbit_ndot: ballistic coefficient (revs/day)
        orbit_nddot: mean motion 2nd derivative (revs/day^3)
        orbit_periapsis: perigee (m)
        orbit_apoapsis: apogee (m)
        orbital_precession: precessoin due to J2 (rad/s)
        orbit_period: period (s)

    Args:
        name: Name of a satellite
        tsync: Datetime object that gives the date and time at simulation time 0

    """

    __slots__ = []

    @staticmethod
    def getInitialEpoch() -> datetime:
        """Return the inital epoch of TLEs : 1949 December 31 00:00 UT

        Returns:
            float: Inital epoch of TLEs (s)

        """
        t0 = datetime(year=1949, month=12, day=31, hour=0, minute=0, second=0, tzinfo=timezone.utc)
        return t0

    def __init__(self, name: str, tsync: datetime):
        AComputer.__init__(self, name=name)

        self.defineOutput(
            name="itrf", snames=["px", "py", "pz", "vx", "vy", "vz"], dtype=np.float64
        )
        self.defineOutput(name="subpoint", snames=["lon", "lat"], dtype=np.float64)

        self.createParameter(name="tsync", value=tsync, read_only=True)
        self.createParameter(name="orbit_mano")  # (rad)
        self.createParameter(name="orbit_eccentricity")
        self.createParameter(name="orbit_semi_major_axis")  # (m)
        self.createParameter(name="orbit_inclination")  # (rad)
        self.createParameter(name="orbit_argp")  # (rad)
        self.createParameter(name="orbit_node")  # (rad)
        self.createParameter(name="orbit_bstar")  # (/earth radii)
        self.createParameter(name="orbit_ndot")  # (revs/day)
        self.createParameter(name="orbit_nddot")  # (revs/day^3)
        self.createParameter(name="orbit_periapsis")  # (m)
        self.createParameter(name="orbit_apoapsis")  # (m)
        self.createParameter(name="orbital_precession")  # (rad/s)
        self.createParameter(name="orbit_period")  # (s)

    def subpoint(self, itrf_pos_vel: NDArray[Any, Any]) -> Tuple[float, float]:
        """
        Return the longitude and latitude directly beneath this position.

        Args:
            itrf_pos_vel: ITRF position (m) and velocity (m/s)

        Returns:
            A tuple lon, lat (rad)

        """
        lon, lat, _ = itrf_to_geodetic(itrf_pos_vel)

        return lon, lat

    def geocentricITRFTrajectory(
        self,
        number_of_position: int = 200,
        number_of_periods: int = 1,
        color=(1, 0, 0, 1),
    ) -> Trajectory:
        """
        Return the geocentric ITRF positions of the trajectory

        Args:
            number_of_position: Number of points per orbital period
            number_of_periods: Number of orbit periods to plot
            color: The color as a 4-elements tuple:

            *  r between 0 and 1
            *  g between 0 and 1
            *  b between 0 and 1
            *  alpha between 0 and 1 (use 1 for fully opaque)

        Returns:
            blocksim.satellite.Trajectory: The generated trajectory

        """
        Ts = self.orbit_period.total_seconds()
        dt = number_of_periods * Ts / number_of_position

        x = np.empty(number_of_position)
        y = np.empty(number_of_position)
        z = np.empty(number_of_position)

        for i in range(number_of_position):
            t = i * dt
            output = self.update(t - dt, t, itrf=None, subpoint=None)
            x[i], y[i], z[i], _, _, _ = output["itrf"]

        traj = Trajectory(name=self.getName(), x=x, y=y, z=z, color=color)

        return traj

    def update(
        self, t1: float, t2: float, subpoint: NDArray[Any, Any], itrf: NDArray[Any, Any]
    ) -> dict:
        outputs = {}
        outputs["itrf"] = self.getGeocentricITRFPositionAt(t2)
        outputs["subpoint"] = np.array(self.subpoint(outputs["itrf"]))

        return outputs

    def getTrajectoryFromLogger(self, log: "Logger", color=(1, 0, 0, 1)) -> Trajectory:
        """Bulids a Trajectory for the ASatellite from a Logger

        Args:
            log: The logger to read
            color: The color as a 4-elements tuple:

            *  r between 0 and 1
            *  g between 0 and 1
            *  b between 0 and 1
            *  alpha between 0 and 1 (use 1 for fully opaque)

        Returns:
            A Trajectory instance

        """
        traj = Trajectory.fromLogger(
            log,
            name=self.getName(),
            params=(
                "t",
                f"{self.getName()}_itrf_px",
                f"{self.getName()}_itrf_py",
                f"{self.getName()}_itrf_pz",
            ),
            color=color,
        )
        return traj

    @abstractmethod
    def getGeocentricITRFPositionAt(self, td: float) -> NDArray[Any, Any]:  # pragma: no cover
        """Abstract method that shall compute, for a simulation time td,
        an array with 3 cartesian position (m) and 3 cartesian velocity (m/s) in ITRF frame

        Args:
            td: Simulation time (s)

        Returns:
            array: A 6-elements array with 3 position scalars (m) and 3 velocity scalar (m/s) in ITRF frame

        """
        pass


class SGP4Satellite(ASatellite):
    """Earth-orbiting satellite, using SGP4 propagator

    The outputs of the computer are **itrf** and **subpoint**

    itrf:

    * px: X coordinate in geocentric ITRF (m)
    * py: Y coordinate in geocentric ITRF (m)
    * pz: Z coordinate in geocentric ITRF (m)
    * vx: X coordinate of velocity in geocentric ITRF (m/s)
    * vy: Y coordinate of velocity in geocentric ITRF (m/s)
    * vz: Z coordinate of velocity in geocentric ITRF (m/s)

    subpoint:

    * lon: Longitude (rad)
    * lat: Latitude (rad)

    Args:
        name: Name of the element

    """

    __slots__ = ["__sgp4"]

    def __init__(self, name: str, tsync: datetime):
        ASatellite.__init__(self, name, tsync)
        self.__sgp4 = None

    def __setSGP4(self, sgp4):
        self.__sgp4 = sgp4
        self.orbit_mano = sgp4.mo
        self.orbit_eccentricity = sgp4.ecco
        self.orbit_inclination = sgp4.inclo
        self.orbit_argp = sgp4.argpo
        self.orbit_node = sgp4.nodeo
        self.orbit_bstar = sgp4.bstar
        self.orbit_ndot = sgp4.ndot
        self.orbit_nddot = sgp4.nddot
        n = sgp4.no_kozai / 60
        self.orbit_period = timedelta(seconds=2 * np.pi / n)
        inc = self.orbit_inclination
        e = self.orbit_eccentricity
        ws = 2 * np.pi / self.orbit_period.total_seconds()
        a = (mu / ws**2) ** (1 / 3)
        e = self.orbit_eccentricity
        # https://en.wikipedia.org/wiki/Nodal_precession#Rate_of_precession
        self.orbital_precession = -3 / 2 * (Req / (a * (1 - e**2))) ** 2 * J2 * ws * np.cos(inc)
        self.orbit_semi_major_axis = a
        self.orbit_apoapsis = a * (1 + e)
        self.orbit_periapsis = a * (1 - e)

        pv0 = self.getGeocentricITRFPositionAt(0)
        otp = self.getOutputByName("itrf")
        otp.setInitialState(pv0)

    def getGeocentricITRFPositionAt(self, t_calc: float) -> NDArray[Any, Any]:
        # epoch time in days from jan 0, 1950. 0 hr
        dt = (self.tsync - self.getInitialEpoch()).total_seconds() + t_calc

        pv_teme = sgp4_to_teme(self.__sgp4, dt)
        pv_itrf = teme_to_itrf(dt, pv_teme)

        return pv_itrf

    @classmethod
    def fromOrbitalElements(
        cls,
        name: str,
        tsync: datetime,
        a: float,
        ecc: float,
        argp: float,
        inc: float,
        mano: float,
        node: float,
        bstar: float = 0,
        ndot: float = 0,
        nddot: float = 0,
    ) -> "blocksim.satellite.Satellite.SGP4Satellite":
        """Builds a SGP4Satellite from orbital elements.
        See https://rhodesmill.org/skyfield/earth-satellites.html#build-a-satellite-from-orbital-elements

        Args:
            name: Name of the element
            t: The time of the orbit description
            a: Semi-major axis (m)
            ecc: Eccentricity
            argp: Argument of perigee (rad)
            inc: Inclination (rad)
            mano: Mean anomaly (rad)
            node: Right ascension of ascending node (rad)
            bstar: Drag Term aka Radiation Pressure Coefficient
            ndot: First Derivative of Mean Motion aka the Ballistic Coefficient
            nddot: Second Derivative of Mean Motion

        Returns:
            A SGP4Satellite instance

        """
        t0 = cls.getInitialEpoch()
        epoch = (tsync - t0).total_seconds() / 86400
        n = np.sqrt(mu / a**3)

        # https://rhodesmill.org/skyfield/earth-satellites.html#build-a-satellite-from-orbital-elements
        satrec = Satrec()
        satrec.sgp4init(
            WGS84,  # gravity model
            "i",  # 'a' = old AFSPC mode, 'i' = improved mode
            5,  # satnum: Satellite number
            epoch,  # epoch: days since 1949 December 31 00:00 UT.
            bstar,  # bstar: drag coefficient (/earth radii)
            ndot,  # ndot: ballistic coefficient (revs/day)
            nddot,  # nddot: second derivative of mean motion (revs/day^3)
            ecc,  # eccentricity
            argp,  # argument of perigee (radians)
            inc,  # inclination (radians)
            mano,  # mean anomaly (radians)
            n * 60,  # no_kozai: mean motion (radians/minute)
            node,  # nodeo: right ascension of ascending node (radians)
        )
        sat = cls(name, tsync)
        sat.__setSGP4(satrec)

        return sat

    @classmethod
    def fromTLE(
        cls,
        tle_file: Union[str, Path],
        tsync: datetime = None,
        iline: int = 0,
        name_prefix: str = "",
    ) -> "blocksim.satellite.Satellite.SGP4Satellite":
        """Builds a SGP4Satellite from a TLE file
        Returns None if the file was incorrect
        See https://en.wikipedia.org/wiki/Two-line_element_set

        Args:
            tsync: Time that corresponds to the simulation time zero.
                If None, tsync is the time of the TLE
            tle_file: TLE file path or URL
            iline: Number of the object in the TLE (in case of a multi object TLE file)
            name_prefix: Prefix to use to modify the satellites' name

        Returns:
            A SGP4Satellite instance

        """
        # ISS (ZARYA)
        # 1 25544U 98067A   21076.49742957  .00000086  00000-0  97467-5 0  9995
        # 2 25544  51.6441  76.2242 0003393 119.8379  30.2224 15.48910580274380
        tle_file = str(tle_file)
        if tle_file.startswith("https://") or tle_file.startswith("http://"):
            myfile = requests.get(tle_file, allow_redirects=True)
            data = myfile.text
        else:
            with open(tle_file, "r") as f:
                data = f.read()

        lines = []
        read_lines = data.split("\n")
        for l in read_lines:
            if not l.startswith("#"):
                lines.append(l)

        if iline * 3 + 2 >= len(lines):
            return None

        name = lines[iline * 3].strip().split(" ")[0].lower()
        name = name.replace("_", "")
        name = name.replace("-", "")
        if name == "":
            return None
        line1 = lines[iline * 3 + 1]
        line2 = lines[iline * 3 + 2]

        satrec = Satrec.twoline2rv(line1, line2)

        if tsync is None:
            month, day, hour, minute, second = days2mdhms(
                2000 + satrec.epochyr, satrec.epochdays, round_to_microsecond=9
            )
            tsync = datetime(
                year=2000 + satrec.epochyr,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=int(second),
                microsecond=int(1e6 * (second - int(second))),
                tzinfo=timezone.utc,
            )

        sat = cls(name_prefix + name, tsync)
        sat.__setSGP4(satrec)

        return sat

    def toEquinoctialOrbit(self) -> Tuple[float]:
        """

        See https://www.orekit.org/static/apidocs/org/orekit/orbits/EquinoctialOrbit.html

        """
        a = self.orbit_semi_major_axis
        ex = self.orbit_eccentricity * cos(self.orbit_argp + self.orbit_node)
        ey = self.orbit_eccentricity * sin(self.orbit_argp + self.orbit_node)
        hx = tan(self.orbit_inclination / 2) * cos(self.orbit_node)
        hy = tan(self.orbit_inclination / 2) * sin(self.orbit_node)
        lv = self.orbit_mano + self.orbit_argp + self.orbit_node

        return a, ex, ey, hx, hy, lv

    @classmethod
    def fromEquinoctialOrbit(
        cls,
        name: str,
        tsync: datetime,
        a: float,
        ex: float,
        ey: float,
        hx: float,
        hy: float,
        lv: float,
    ) -> "blocksim.satellite.Satellite.SGP4Satellite":
        """Builds a SGP4Satellite from its equinoctial elements

        See https://www.orekit.org/static/apidocs/org/orekit/orbits/EquinoctialOrbit.html

        Returns:
            A SGP4Satellite instance

        """
        ecc = lin.norm([ex, ey])
        Om = np.arctan2(hy, hx)
        wPOm = np.arctan2(ey, ex)
        w = wPOm - Om
        tan_inc_2 = lin.norm([hx, hy])
        inc = 2 * np.arctan(tan_inc_2)
        mano = lv - wPOm

        sat = cls.fromOrbitalElements(
            name=name,
            tsync=tsync,
            a=a,
            ecc=ecc,
            argp=w,
            inc=inc,
            mano=mano,
            node=Om,
            bstar=0,
            ndot=0,
            nddot=0,
        )

        return sat


class CircleSatellite(ASatellite):
    """Earth-orbiting satellite, which follow a circle

    The outputs of the computer are **itrf** and **subpoint**

    itrf:

    * px: X coordinate in geocentric ITRF (m)
    * py: Y coordinate in geocentric ITRF (m)
    * pz: Z coordinate in geocentric ITRF (m)
    * vx: X coordinate of velocity in geocentric ITRF (m/s)
    * vy: Y coordinate of velocity in geocentric ITRF (m/s)
    * vz: Z coordinate of velocity in geocentric ITRF (m/s)

    subpoint:

    * lon: Longitude (rad)
    * lat: Latitude (rad)

    Args:
        name: Name of the element
        tsync: Datetime object that gives the date and time at simulation time 0

    """

    __slots__ = ["__sat_puls", "__R1", "__R2"]

    def __init__(self, name: str, tsync: datetime):
        ASatellite.__init__(self, name, tsync)

    def setInitialITRF(self, t_epoch: float, pv: NDArray[Any, Any]):
        """Set the initial position and velocity from in ITRF position / velocity
        The velocity is used only to determine the orbit's plane. It is then modified so that the orbit eccentricity be 0
        Also sets the attributes of the class

        Args:
            t_epoch: Time since 31/12/1949 00:00 UT (s)
            pv: A 6-elements array with 3 position scalars (m) and 3 velocity scalar (m/s) in ITRF frame

        """
        pv_teme = itrf_to_teme(t_epoch=t_epoch, pv_itrf=pv)
        pos = pv_teme[:3]
        vel = pv_teme[3:]
        a = lin.norm(pos)
        self.__sat_puls = sqrt(mu / a**3)

        n = np.cross(pos, vel)
        n /= lin.norm(n)
        vel = np.cross(n, pos) * self.__sat_puls
        pvc_teme = np.hstack((pos, vel))

        self.__R1 = pvc_teme
        self.__R2 = np.hstack((np.cross(n, pos), np.cross(n, vel)))

        pvc = teme_to_itrf(t_epoch=t_epoch, pv_teme=pvc_teme)
        self.setInitialStateForOutput(initial_state=pvc, output_name="itrf")
        self.createParameter(name="initial_itrf", value=pv, read_only=True)

        a, e, argp, inc, mano, node = teme_to_orbital(pvc_teme)

        self.orbit_mano = mano
        self.orbit_eccentricity = e
        self.orbit_inclination = inc
        self.orbit_argp = argp
        self.orbit_node = node
        self.orbit_bstar = 0.0
        self.orbit_ndot = 0.0
        self.orbit_nddot = 0.0
        ws = sqrt(mu / a**3)
        self.orbit_period = timedelta(seconds=2 * np.pi / ws)
        # https://en.wikipedia.org/wiki/Nodal_precession#Rate_of_precession
        self.orbital_precession = -3 / 2 * (Req / (a * (1 - e**2))) ** 2 * J2 * ws * np.cos(inc)
        self.orbit_semi_major_axis = a
        self.orbit_apoapsis = a * (1 + e)
        self.orbit_periapsis = a * (1 - e)

        pv0 = self.getGeocentricITRFPositionAt(0)
        otp = self.getOutputByName("itrf")
        otp.setInitialState(pv0)

    @classmethod
    def fromOrbitalElements(
        cls,
        name: str,
        tsync: datetime,
        a: float,
        inc: float,
        argp: float,
        mano: float = 0.0,
        node: float = 0.0,
    ) -> "blocksim.satellite.Satellite.CircleSatellite":
        """Builds an orbit from orbital elements. The eccentricity is forced to 0

        Args:
            name: Name of the satellite
            tsync: Date that corresponds to simulation time = 0
            a: Semi-major axis (m)
            inc: Inclination (rad)
            argp: Argument of periapsis (rad)
            mano: Mean anomaly (rad)
            node: Longitude of the ascending node (rad)

        Returns:
            A CircleSatellite instance

        """
        t_epoch = (tsync - ASatellite.getInitialEpoch()).total_seconds()
        pv_teme = orbital_to_teme(a=a, ecc=0, argp=argp, inc=inc, mano=mano, node=node)
        pv_itrf = teme_to_itrf(t_epoch=t_epoch, pv_teme=pv_teme)

        sat = CircleSatellite.fromITRF(name=name, tsync=tsync, pv_itrf=pv_itrf)

        sat.createParameter("a", value=a, read_only=True)
        sat.createParameter("inc", value=inc, read_only=True)
        sat.createParameter("argp", value=argp, read_only=True)
        sat.createParameter("mano", value=mano, read_only=True)
        sat.createParameter("node", value=node, read_only=True)

        return sat

    @classmethod
    def fromITRF(
        cls, name: str, tsync: datetime, pv_itrf: NDArray[Any, Any]
    ) -> "blocksim.satellite.Satellite.CircleSatellite":
        """Instanciates a CircleSatellite from an initial position and velocity from in ITRF position / velocity
        The velocity is used only to determine the orbit's plane. It is then modified so that the orbit eccentricity be 0
        Also sets the attributes of the class

        Args:
            tsync: Time that corresponds to the simulation time zero
            pv_itrf: A 6-elements array with 3 position scalars (m) and 3 velocity scalar (m/s) in ITRF frame

        Returns:
            A CircleSatellite instance

        """
        sat = cls(name, tsync)

        t_epoch = (tsync - ASatellite.getInitialEpoch()).total_seconds()
        sat.setInitialITRF(t_epoch=t_epoch, pv=pv_itrf)

        return sat

    @classmethod
    def fromTLE(
        cls,
        tle_file: Union[str, Path],
        tsync: datetime = None,
        iline: int = 0,
        name_prefix: str = "",
    ) -> "blocksim.satellite.Satellite.CircleSatellite":
        """Builds a CircleSatellite from a TLE file
        The velocity is used only to determine the orbit's plane. It is then modified so that the orbit eccentricity be 0
        Returns None if the file was incorrect
        See https://en.wikipedia.org/wiki/Two-line_element_set

        Args:
            tsync: Time that corresponds to the simulation time zero.
                If None, tsync is the time of the TLE
            tle_file: TLE file path or URL
            iline: Number of the object in the TLE (in case of a multi object TLE file)
            name_prefix: Prefix to use to modify the satellites' name

        Returns:
            A CircleSatellite instance

        """
        # ISS (ZARYA)
        # 1 25544U 98067A   21076.49742957  .00000086  00000-0  97467-5 0  9995
        # 2 25544  51.6441  76.2242 0003393 119.8379  30.2224 15.48910580274380
        tle_file = str(tle_file)
        if tle_file.startswith("https://") or tle_file.startswith("http://"):
            myfile = requests.get(tle_file, allow_redirects=True, verify=False)
            data = myfile.text
        else:
            with open(tle_file, "r") as f:
                data = f.read()

        lines = []
        read_lines = data.split("\n")
        for l in read_lines:
            if not l.startswith("#"):
                lines.append(l)

        if iline * 3 + 2 >= len(lines):
            return None

        name = lines[iline * 3].strip().split(" ")[0].lower()
        name = name.replace("_", "")
        name = name.replace("-", "")
        if name == "":
            return None
        line1 = lines[iline * 3 + 1]
        line2 = lines[iline * 3 + 2]

        satrec = Satrec.twoline2rv(line1, line2)

        if tsync is None:
            month, day, hour, minute, second = days2mdhms(
                2000 + satrec.epochyr, satrec.epochdays, round_to_microsecond=9
            )
            tsync = datetime(
                year=2000 + satrec.epochyr,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=int(second),
                microsecond=int(1e6 * (second - int(second))),
                tzinfo=timezone.utc,
            )

        dt = (tsync - ASatellite.getInitialEpoch()).total_seconds()
        pv_teme = sgp4_to_teme(satrec, dt)
        pv_itrf = teme_to_itrf(dt, pv_teme)

        sat = CircleSatellite.fromITRF(name=name_prefix + name, tsync=tsync, pv_itrf=pv_itrf)

        return sat

    def getGeocentricITRFPositionAt(self, td: float) -> NDArray[Any, Any]:
        t_epoch = (self.tsync - self.getInitialEpoch()).total_seconds() + td

        th = self.__sat_puls * td

        cth = cos(th)
        sth = sin(th)

        # if hasattr(td, "__iter__"):
        #     ns = len(td)
        #     newpv_teme = np.outer(self.__R1, cth) + np.outer(self.__R2, sth)
        #     newpv = np.empty_like(newpv_teme)
        #     for k in range(ns):
        #         newpv[:, k] = teme_to_itrf(t_epoch=t_epoch, pv_teme=newpv_teme[:, k])
        # else:
        newpv_teme = self.__R1 * cth + self.__R2 * sth
        newpv = teme_to_itrf(t_epoch=t_epoch, pv_teme=newpv_teme)

        return newpv


def createSatellites(
    tle_file: str,
    tsync: datetime,
    prop: ASatellite = SGP4Satellite,
    name_prefix: str = "",
    sim: Simulation = None,
    connect_to: str = "tkr",
) -> List[ASatellite]:
    """Creates a list of satellites from a TLE file, using the specified subclass of ASatellite
    If sim is not None and connect_to is not "", the created satellites will be:
    * inserted in sim
    * grouped in a Group AComputer
    * connected together to the GNSSTracker

    Args:
        tle_file: TLE file path or URL
        tsync: Time that corresponds to the simulation time zero
        prop: Propagator to use, as a subclass of ASatellite
        name_prefix: Prefix to use to modify the satellites' name
        sim: Instance of a Simulation
        connect_to: Name of the GNSSTracker to use in the simulation

    Returns:
        A list of instances of ASatellite (following prop argument),
            one per object described in tle_file

    """
    iline = 0
    satellites = []
    while True:
        sat = prop.fromTLE(tsync=tsync, tle_file=tle_file, iline=iline, name_prefix=name_prefix)
        if sat is None:
            break
        sat.getName
        satellites.append(sat)
        iline += 1

    nom_coord = ["px", "py", "pz", "vx", "vy", "vz"]
    if not sim is None and not connect_to == "":
        grp_snames = []
        grp_inp = dict()
        for k, sat in enumerate(satellites):
            sim.addComputer(sat)

            grp_inp["itrf%i" % k] = (6,)
            grp_snames.extend(["%s%i" % (n, k) for n in nom_coord])

        grp = Group(
            "grp",
            inputs=grp_inp,
            snames=grp_snames,
        )

        sim.addComputer(grp)

        for k, sat in enumerate(satellites):
            sim.connect("%s.itrf" % sat.getName(), "grp.itrf%i" % k)

        sim.connect("grp.grouped", "tkr.state")

    return satellites
