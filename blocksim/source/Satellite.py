from abc import abstractmethod
from typing import Tuple, List
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy import cos, sin, tan, sqrt, pi
import scipy.linalg as lin
from skyfield.api import EarthSatellite, load, utc
from sgp4.api import Satrec, WGS84
from skyfield.units import Distance, Velocity
from skyfield.positionlib import Geocentric
from skyfield.sgp4lib import TEME_to_ITRF

from .. import logger
from ..constants import *
from ..utils import datetime_to_skyfield, skyfield_to_datetime, itrf_to_geodetic
from ..core.Node import AComputer
from .Trajectory import Trajectory


__all__ = ["ASatellite", "CircleSatellite", "Satellite"]


class ASatellite(AComputer):

    __slots__ = []

    @staticmethod
    def getInitialEpoch() -> datetime:
        """

        Return the inital epoch of TLEs : 1949 December 31 00:00 UT

        Returns:
          Inital epoch of TLEs

        """
        t0 = datetime(
            year=1949, month=12, day=31, hour=0, minute=0, second=0, tzinfo=timezone.utc
        )
        return t0

    def __init__(self, name: str):
        AComputer.__init__(self, name=name)

        self.defineOutput(
            name="itrf", snames=["px", "py", "pz", "vx", "vy", "vz"], dtype=np.float64
        )

    def subpoint(self, td: float) -> Tuple[float]:
        """
        Return the latitude and longitude directly beneath this position.

        Args:
          td (s)
            Time elapsed since initial time

        Returns:
          lon, lat (rad)

        """
        itrf_pos_vel = self.getGeocentricITRFPositionAt(td)

        lon, lat, _ = itrf_to_geodetic(itrf_pos_vel)

        return lon, lat

    def geocentricITRFTrajectory(
        self, number_of_position=200, number_of_periods=1, color=(1, 0, 0, 0)
    ) -> Trajectory:
        """
        Return the geocentric ITRF positions of the trajectory

        Args:
          number_of_position
            Number of points per orbital period
          number_of_periods
            Number of orbit periods to plot
          color
            The color as a 4-elements tuple:
            r between 0 and 1
            g between 0 and 1
            b between 0 and 1
            alpha between 0 and 1

        Returns:
          Trajectory

        """
        Ts = self.orbit_period.total_seconds()
        dt = number_of_periods * Ts / number_of_position

        x = np.empty(number_of_position)
        y = np.empty(number_of_position)
        z = np.empty(number_of_position)

        for i in range(number_of_position):
            t = i * dt
            output = self.compute_outputs(t - dt, t, itrf=None, subpoint=None)
            x[i], y[i], z[i], _, _, _ = output["itrf"]

        traj = Trajectory(name=self.getName(), x=x, y=y, z=z, color=color)

        return traj

    @abstractmethod
    def getGeocentricITRFPositionAt(self, td: float) -> np.array:
        pass


class Satellite(ASatellite):
    """Earth-orbiting satellite, using SGP4 propagator

    The output name of the computer are **itrf** and **subpoint**

    Args:
      name
        Name of the element

    itrf:

    * px : X coordinate in geocentric ITRF (m)
    * py : Y coordinate in geocentric ITRF (m)
    * pz : Z coordinate in geocentric ITRF (m)
    * vx : X coordinate of velocity in geocentric ITRF (m/s)
    * vy : Y coordinate of velocity in geocentric ITRF (m/s)
    * vz : Z coordinate of velocity in geocentric ITRF (m/s)

    subpoint:

    * lon : Longitude (rad)
    * lat : Latitude (rad)

    """

    __slots__ = ["__sgp4", "__epoch"]

    def __init__(self, name: str):
        ASatellite.__init__(self, name)
        self.defineOutput(name="subpoint", snames=["lon", "lat"], dtype=np.float64)
        self.__sgp4 = None
        self.createParameter("tsync", value=None)

    def _setSGP4(self, epoch: datetime, sgp4: Satrec):
        self.__sgp4 = sgp4
        self.__epoch = epoch

    def getGeocentricITRFPositionAt(self, t_calc: float) -> np.array:
        """
        Return the geocentric ITRF position of the satellite at a given time

        Args:
          t_calc (s)
            Time elapsed since initial time

        Returns:
          x, y, z (m)
          vx, vy, vz (m/s)

        """
        # epoch time in days from jan 0, 1950. 0 hr
        dt = (self.tsync - self.getInitialEpoch()).total_seconds() + t_calc
        epoch = dt / 86400

        whole, fraction = divmod(epoch, 1.0)
        whole_jd = whole + 2433281.5

        jd = whole_jd
        fraction = fraction
        e, rTEME, vTEME = self.__sgp4.sgp4(jd, fraction)
        if e == 1:
            raise AssertionError("Mean eccentricity is outside the range 0 ≤ e < 1.")
        elif e == 2:
            raise AssertionError("Mean motion has fallen below zero.")
        elif e == 3:
            raise AssertionError(
                "Perturbed eccentricity is outside the range 0 ≤ e ≤ 1."
            )
        elif e == 4:
            raise AssertionError(
                "Length of the orbit’s semi-latus rectum has fallen below zero."
            )
        elif e == 6:
            raise AssertionError(
                "Orbit has decayed: the computed position is underground. (The position is still returned, in case the vector is helpful to software that might be searching for the moment of re-entry.)"
            )

        rTEME = np.array(rTEME)  # km
        vTEME = np.array(vTEME)  # km/s

        rTEME /= AU_KM
        vTEME /= AU_KM
        vTEME *= DAY_S

        rITRF, vITRF = TEME_to_ITRF(jd, rTEME, vTEME, 0.0, 0.0, fraction)

        pv = np.empty(6)
        pv[:3] = rITRF * AU_M
        pv[3:] = vITRF * AU_M / DAY_S
        return pv

    def compute_outputs(
        self, t1: float, t2: float, subpoint: np.array, itrf: np.array
    ) -> dict:
        outputs = {}
        outputs["itrf"] = self.getGeocentricITRFPositionAt(t2)
        outputs["subpoint"] = np.array(self.subpoint(t2))

        return outputs

    @classmethod
    def fromOrbitalElements(
        cls,
        name: str,
        t: datetime,
        a: float,
        ecc: float,
        argp: float,
        inc: float,
        mano: float,
        node: float,
        bstar: float = 0,
        ndot: float = 0,
        nddot: float = 0,
    ) -> "Satellite":
        """Builds a Satellite from orbital elements.
        See https://rhodesmill.org/skyfield/earth-satellites.html#build-a-satellite-from-orbital-elements

        Args:
          name
            Name of the element
          t : a datetime instance
            The time of the orbit description
          a (m)
            Semi-major axis
          ecc
            Eccentricity
          argp (rad)
            Argument of perigee
          inc (rad)
            Inclination
          mano (rad)
            Mean anomaly
          node (rad)
            Right ascension of ascending node
          bstar
            Drag Term aka Radiation Pressure Coefficient
          ndot
            First Derivative of Mean Motion aka the Ballistic Coefficient
          nddot
            Second Derivative of Mean Motion

        """
        t0 = cls.getInitialEpoch()
        epoch = (t - t0).total_seconds() / 86400
        n = np.sqrt(mu / a ** 3)

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
        sat = cls(name)
        sat._setSGP4(t, satrec)

        sat.tsync = t

        return sat

    @classmethod
    def fromTLE(cls, tle_file: str, iline: int = 0) -> "Satellite":
        """Builds a Satellite from a TLE file?
        Returns None if the file was incorrect
        See https://en.wikipedia.org/wiki/Two-line_element_set

        Args:
          tle_file
            TLE file path
          iline
            Number of the object in the TLE (in case of a multi object TLE file)

        """
        # ISS (ZARYA)
        # 1 25544U 98067A   21076.49742957  .00000086  00000-0  97467-5 0  9995
        # 2 25544  51.6441  76.2242 0003393 119.8379  30.2224 15.48910580274380
        lines = []
        with open(tle_file, "r") as f:
            while True:
                l = f.readline()
                if l == "":
                    break
                if l[0] != "#":
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
        two_digit_year = satrec.epochyr
        if two_digit_year < 57:
            year = two_digit_year + 2000
        else:
            year = two_digit_year + 1900

        ts = load.timescale()
        epoch = skyfield_to_datetime(ts.utc(year, 1, satrec.epochdays))

        sat = cls(name)
        sat._setSGP4(epoch, satrec)

        sat.tsync = epoch

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
        t: datetime,
        a: float,
        ex: float,
        ey: float,
        hx: float,
        hy: float,
        lv: float,
    ) -> "Satellite":
        """Builds a Satellite from its equinoctial elements

        See https://www.orekit.org/static/apidocs/org/orekit/orbits/EquinoctialOrbit.html

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
            t=t,
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

    @property
    def orbit_mano(self) -> float:
        """
        Return the mean anomaly

        Returns:
          mano (rad)

        """
        return self.__sgp4.mo

    @property
    def orbit_eccentricity(self) -> float:
        """
        Return the eccentricity

        Returns:
          ecc

        """
        return self.__sgp4.ecco

    @property
    def orbit_semi_major_axis(self) -> float:
        """
        Return the sami-major axis

        Returns:
          a (m)

        """
        ws = 2 * np.pi / self.orbit_period.total_seconds()
        a = (mu / ws ** 2) ** (1 / 3)
        return a

    @property
    def orbit_inclination(self) -> float:
        """
        Return the inclination

        Returns:
          inc (rad)

        """
        return self.__sgp4.inclo

    @property
    def orbit_argp(self) -> float:
        """
        Return the argument of perigee

        Returns:
          argp (rad)

        """
        return self.__sgp4.argpo

    @property
    def orbit_node(self) -> float:
        """
        Return the right ascension of ascending node

        Returns:
          node (rad)

        """
        return self.__sgp4.nodeo

    @property
    def orbit_bstar(self) -> float:
        """
        Return the drag coefficient

        Returns:
          bstar (/earth radii)

        """
        return self.__sgp4.bstar

    @property
    def orbit_ndot(self) -> float:
        """
        Return the ballistic coefficient

        Returns:
          ndot (revs/day)

        """
        return self.__sgp4.ndot

    @property
    def orbit_nddot(self) -> float:
        """
        Return the second derivative of mean motion

        Returns:
          nddot (revs/day^3)

        """
        return self.__sgp4.nddot

    @property
    def orbit_periapsis(self) -> float:
        """
        Return orbital periapsis

        Returns:
          per (m)

        """
        a = self.orbit_semi_major_axis
        e = self.orbit_eccentricity
        return a * (1 - e)

    @property
    def orbit_apoapsis(self) -> float:
        """
        Return orbital apoapsis

        Returns:
          apo (m)

        """
        a = self.orbit_semi_major_axis
        e = self.orbit_eccentricity
        return a * (1 + e)

    @property
    def orbital_precession(self) -> float:
        """
        Return orbital precession of the orbit due to J2

        Returns:
          w (rad/s)

        """
        # https://en.wikipedia.org/wiki/Nodal_precession#Rate_of_precession
        ws = 2 * np.pi / self.orbit_period.total_seconds()
        inc = self.orbit_inclination
        a = self.orbit_semi_major_axis
        e = self.orbit_eccentricity

        return -3 / 2 * (Req / (a * (1 - e ** 2))) ** 2 * J2 * ws * np.cos(inc)

    @property
    def orbit_period(self) -> timedelta:
        """
        Return the period of the orbit

        Returns:
          T (s)

        """
        # https://en.wikipedia.org/wiki/Mean_motion#Mean_motion_and_Kepler's_laws
        n = self.__sgp4.no_kozai / 60
        return timedelta(seconds=2 * np.pi / n)

    @property
    def epoch(self) -> datetime:
        """
        Return the epoch of the orbit

        Returns:
          Epoch

        """
        return self.__epoch


class CircleSatellite(ASatellite):
    """Earth-orbiting satellite, which follow a circle

    The output name of the computer is **itrf**

    Args:
      name
        Name of the element

    itrf:

    * px : X coordinate in geocentric ITRF (m)
    * py : Y coordinate in geocentric ITRF (m)
    * pz : Z coordinate in geocentric ITRF (m)
    * vx : X coordinate of velocity in geocentric ITRF (m/s)
    * vy : Y coordinate of velocity in geocentric ITRF (m/s)
    * vz : Z coordinate of velocity in geocentric ITRF (m/s)

    Args:
      name
        Name of the element
      t : a datetime instance
        The time of the orbit description
      a (m)
        Semi-major axis
      inc (rad)
        Inclination
      mano (rad)
        Mean anomaly
      node (rad)
        Right ascension of ascending node

    """

    __slots__ = ["__sat_puls", "__R1", "__R2"]

    def __init__(
        self,
        name: str,
        t: datetime,
        a: float,
        inc: float,
        mano: float = 0.0,
        node: float = 0.0,
    ):
        ASatellite.__init__(self, name)

        t0 = self.getInitialEpoch()
        epoch = (t - t0).total_seconds() / 86400
        n = np.sqrt(mu / a ** 3)

        # https://rhodesmill.org/skyfield/earth-satellites.html#build-a-satellite-from-orbital-elements
        satrec = Satrec()
        satrec.sgp4init(
            WGS84,  # gravity model
            "i",  # 'a' = old AFSPC mode, 'i' = improved mode
            5,  # satnum: Satellite number
            epoch,  # epoch: days since 1949 December 31 00:00 UT.
            0.0,  # bstar: drag coefficient (/earth radii)
            0.0,  # ndot: ballistic coefficient (revs/day)
            0.0,  # nddot: second derivative of mean motion (revs/day^3)
            0.0,  # eccentricity
            0.0,  # argument of perigee (radians)
            inc,  # inclination (radians)
            mano,  # mean anomaly (radians)
            n * 60,  # no_kozai: mean motion (radians/minute)
            node,  # nodeo: right ascension of ascending node (radians)
        )
        e, rTEME, vTEME = satrec.sgp4(satrec.jdsatepoch, satrec.jdsatepochF)
        assert e == 0

        rTEME = np.array(rTEME)  # km
        vTEME = np.array(vTEME)  # km/s

        rTEME /= AU_KM
        vTEME /= AU_KM
        vTEME *= DAY_S

        rITRF, vITRF = TEME_to_ITRF(
            satrec.jdsatepoch, rTEME, vTEME, 0.0, 0.0, satrec.jdsatepochF
        )
        pos = rITRF * AU_M
        vel = vITRF * AU_M / DAY_S

        pv = np.empty(6)
        pv[:3] = pos
        pv[3:] = vel

        self.createParameter("t", value=t, read_only=True)
        self.createParameter("a", value=a, read_only=True)
        self.createParameter("inc", value=inc, read_only=True)
        self.createParameter("mano", value=mano, read_only=True)
        self.createParameter("node", value=node, read_only=True)
        self.createParameter(name="initial_itrf", value=pv, read_only=True)

        self.setInitialStateForOutput(initial_state=pv, output_name="itrf")

        n = np.cross(pos, vel)
        n /= lin.norm(n)

        # Precomputed termes of the Rodrigues's formula
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
        self.__R1 = pv
        self.__R2 = np.hstack((np.cross(n, pos), np.cross(n, vel)))

        self.__sat_puls = sqrt(mu / a ** 3)

    def getGeocentricITRFPositionAt(self, td: float) -> np.array:
        th = self.__sat_puls * td

        cth = cos(th)
        sth = sin(th)

        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
        newpv = self.__R1 * cth + self.__R2 * sth

        return newpv

    def compute_outputs(self, t1: float, t2: float, itrf: np.array) -> dict:
        outputs = {}
        outputs["itrf"] = self.getGeocentricITRFPositionAt(t2)

        return outputs


def createSatellites(tle_file: str, tsync: datetime) -> List[Satellite]:
    iline = 0
    satellites = []
    while True:
        sat = Satellite.fromTLE(tle_file, iline=iline)
        if sat is None:
            break
        sat.tsync = tsync
        satellites.append(sat)
        iline += 1

    return satellites
