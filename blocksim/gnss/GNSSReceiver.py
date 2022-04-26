from typing import List, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy import sqrt
import scipy.linalg as lin
from scipy.optimize import minimize, Bounds
from skyfield.api import Topos, load
from skyfield import framelib

from ..core.Node import AComputer, Output

from .. import logger
from ..utils import datetime_to_skyfield, skyfield_to_datetime, build_env
from ..constants import c as clum
from ..utils import pdot


class UEPositionOutput(Output):

    __slots__ = ["__sgp4_rep", "__tsync"]

    def __init__(self, name: str, lon: float, lat: float, alt: float, tsync: datetime):
        Output.__init__(
            self, name=name, snames=["x", "y", "z", "vx", "vy", "vz"], dtype=np.float64
        )
        self.__sgp4_rep = Topos(
            latitude_degrees=lat, longitude_degrees=lon, elevation_m=alt
        )
        self.__tsync = tsync

    def getTsync(self) -> datetime:
        return self.__tsync

    def getGeocentricITRFPositionAt(self, t_calc: datetime) -> "array":
        """Return the geocentric ITRF position of the satellite at a given time

        Args:
            td: Time of the position

        Returns:
            A tuple of x, y, z ITRF coordinates (m)

        """
        t = datetime_to_skyfield(t_calc)

        pv = self.__sgp4_rep.at(t)
        pos, vel = pv.frame_xyz_and_velocity(framelib.itrs)

        ps = pos.m
        vs = vel.km_per_s * 1000

        return np.hstack((ps, vs))

    def resetCallback(self, t0: float):
        super().resetCallback(t0)
        dt = timedelta(seconds=t0)
        td = self.__tsync + dt
        state = self.getGeocentricITRFPositionAt(td)
        self.setInitialState(state)


class GNSSReceiver(AComputer):
    """GNSS receiver. Warning: before any simulation, you should call initialisePosition

    The input of the computer is **measurements**
    The outputs of the computer are **realpos**, **estpos** and **estclkerror**

    The following parameters are to be defined by the user :

    Attributes:
        algo: Type of algorithm to use. Can be 'ranging' or 'doppler'
        optim: Defaults to "trust-constr"

    Args:
        name: Name of the element
        nsat: Number of satellites flollowed by the tracker
        lon: True longitude of the receiver (deg)
        lat: True latitude of the receiver (deg)
        alt: True altitude of the receiver (m)
        tsync: Date where the calculation takes place

    """

    __slots__ = []

    def __init__(
        self, name: str, nsat: int, lon: float, lat: float, alt: float, tsync: datetime
    ):
        AComputer.__init__(self, name)
        self.defineInput("measurements", shape=(2 * nsat,), dtype=np.float64)
        self.defineInput("ephemeris", shape=(6 * nsat,), dtype=np.float64)
        otp = UEPositionOutput(name="realpos", lon=lon, lat=lat, alt=alt, tsync=tsync)
        self.addOutput(otp)
        self.defineOutput(
            "estpos", snames=["x", "y", "z", "vx", "vy", "vz"], dtype=np.float64
        )
        self.defineOutput(
            "estdop", snames=["sx", "sy", "sz", "sp", "sv"], dtype=np.float64
        )
        self.defineOutput("estclkerror", snames=["dp", "dv"], dtype=np.float64)

        self.createParameter("algo", value="ranging")
        self.createParameter("optim", value="trust-constr")

    def getTsync(self) -> datetime:
        otp = self.getOutputByName("realpos")
        return otp.getTsync()

    def getSatellitePositionFromEphem(self, ephem: np.array, isat: int) -> "array":
        """Given the array of all satellites ephemeris,
        returns the position for satellite number isat

        Args:
            ephem: The array of all satellites ephemeris
            isat: The number of the considered satellite

        Returns:
            The position of the considered satellite (ITRF, m)

        """
        return ephem[6 * isat : 6 * isat + 3]

    def getSatelliteVelocityFromEphem(self, ephem: np.array, isat: int) -> "array":
        """Given the array of all satellites ephemeris,
        returns the velocity for satellite number isat

        Args:
            ephem: The array of all satellites ephemeris
            isat: The number of the considered satellite

        Returns:
            The velocity of the considered satellite (ITRF, m/s)

        """
        return ephem[6 * isat + 3 : 6 * isat + 6]

    def getPseudorangeFromMeas(self, meas: np.array, isat: int) -> float:
        """Given the array of all measurements,
        returns the pseudo-range for satellite number isat

        Args:
            meas: The array of all measurements
            isat: The number of the considered satellite

        Returns:
            The pseudo-range for the considered satellite (m)

        """
        return meas[2 * isat]

    def getRadialVelocityFromMeas(self, meas: np.array, isat: int) -> float:
        """Given the array of all measurements,
        returns the pseudo-range rate for satellite number isat

        Args:
            meas: The array of all measurements
            isat: The number of the considered satellite

        Returns:
            The pseudo-range rate for the considered satellite (m/s)

        """
        return meas[2 * isat + 1]

    def getDOP(self, ephem: np.array, pv_ue: np.array) -> Tuple[float]:
        """Computes the DOPs

        Args:
            ephem: Ephemeris vector
            pv_ue: UE 3D position/velocity (ITRF) without velocity (m)

        Returns:
            A tuple containing:

            * DOP for X axis (ENV)
            * DOP for Y axis (ENV)
            * DOP for Z axis (ENV)
            * DOP for distance error
            * DOP for velocity error

        """
        pos = pv_ue[:3]
        nsat = len(ephem) // 6
        nval = 0
        if self.algo == "doppler-ranging":
            B = np.empty((2 * nsat, 5))
        else:
            B = np.empty((nsat, 4))

        Penv = build_env(pos)

        for k in range(nsat):
            spos = self.getSatellitePositionFromEphem(ephem, k)
            svel = self.getSatelliteVelocityFromEphem(ephem, k)
            if np.isnan(spos[0]):
                continue

            R = spos - pos
            d = lin.norm(R)

            if self.algo == "doppler":
                B[nval, :3] = -(Penv.T @ svel) / d + (Penv.T @ R) / d**3 * (svel @ R)
                B[nval, 3] = 1
                nval += 1

            elif self.algo == "ranging":
                B[nval, :3] = -(Penv.T @ R) / d
                B[nval, 3] = 1
                nval += 1

            elif self.algo == "doppler-ranging":
                B[nval, :3] = -(Penv.T @ R) / d
                B[nval, 3] = 1
                B[nval, 4] = 0
                nval += 1

                B[nval, :3] = -(Penv.T @ svel) / d + (Penv.T @ R) / d**3 * (svel @ R)
                B[nval, 3] = 0
                B[nval, 4] = 1
                nval += 1

        if nval < 4:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        B = B[:nval, :]

        Q = lin.inv(B.T @ B)

        if self.algo == "doppler":
            sv = sqrt(Q[3, 3])
            sp = 0
        elif self.algo == "ranging":
            sp = sqrt(Q[3, 3])
            sv = 0
        elif self.algo == "doppler-ranging":
            sp = sqrt(Q[3, 3])
            sv = sqrt(Q[4, 4])

        return sqrt(Q[0, 0]), sqrt(Q[1, 1]), sqrt(Q[2, 2]), sp, sv

    def computeFromRadialVelocities(
        self, ephem: np.array, meas: np.array
    ) -> Tuple[np.array, float]:
        """Runs a PVT algorithm that uses only pseudo-range rate

        Args:
            ephem: Array of all ephemeris
            meas: Array of all measurements

        Returns:
            A tuple with:

            * an array of 6 scalar, 3 position (m) and 3 velocity (m/s). Velocities are set to 0
            * range rate bias (m/s)

        """
        nsat = len(meas) // 2

        # On initialise avec la position au nadir du premier satellite visible (ITRF)
        sp0 = np.zeros(3)
        for k in range(nsat):
            spos = self.getSatellitePositionFromEphem(ephem, k)
            if np.isnan(spos[0]):
                continue
            sp0 += spos / nsat
        d0 = lin.norm(sp0)
        pos0 = sp0 / d0 * 6378137
        X0 = np.hstack((pos0, 0))

        mult = np.array([10000000] * 3 + [200])
        # x,y,z,t = X0 + mult*X

        def cout(X):
            Xu = X0 + mult * X
            rpos = Xu[:3]
            dv = Xu[3]

            J = 0

            for k in range(nsat):
                spos = self.getSatellitePositionFromEphem(ephem, k)
                svel = self.getSatelliteVelocityFromEphem(ephem, k)
                radialvelocity = self.getRadialVelocityFromMeas(meas, k)
                if np.isnan(radialvelocity):
                    continue

                ab = spos - rpos
                di = lin.norm(ab)
                vr_est = svel @ ab / di + dv
                err_v = vr_est - radialvelocity

                J += err_v**2

            return J

        def jac(X):
            Xu = X0 + mult * X
            rpos = Xu[:3]
            dv = Xu[3]

            G = np.zeros(4)

            for k in range(nsat):
                spos = self.getSatellitePositionFromEphem(ephem, k)
                svel = self.getSatelliteVelocityFromEphem(ephem, k)
                radialvelocity = self.getRadialVelocityFromMeas(meas, k)
                if np.isnan(radialvelocity):
                    continue

                ab = spos - rpos
                di = lin.norm(ab)
                vr_est = svel @ ab / di + dv
                err_v = vr_est - radialvelocity

                G[:3] += 2 * (-svel / di + ab * (svel @ ab) / di**3) * err_v
                G[3] += 2 * err_v

            return G * mult

        bnd = np.ones(4)

        # jac: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, trust-constr
        # bounds: L-BFGS-B, TNC, SLSQP, Powell, trust-constr
        res = minimize(
            fun=cout,
            jac=jac,
            x0=np.zeros(4),
            method=self.optim,
            # options={"maxfev": 100000, "maxiter": 100000},
            bounds=Bounds(-bnd, bnd),
        )
        if not res.success:
            return None

        Xu = X0 + mult * res.x
        pos = Xu[:3]

        return np.hstack((pos, np.zeros(3))), Xu[3]

    def computeFromPRandVR(
        self, ephem: np.array, meas: np.array
    ) -> Tuple[np.array, float]:
        """Runs a PVT algorithm that uses peudo-range and pseudo-range rate

        Args:
            ephem: Array of all ephemeris
            meas: Array of all measurements

        Returns:
            A tuple with:

            * an array of 6 scalar, 3 position (m) and 3 velocity (m/s). Velocities are set to 0
            * range bias (m)
            * range rate bias (m/s)

        """
        nsat = len(meas) // 2

        # On initialise avec la position au nadir du premier satellite visible (ITRF)
        sp0 = np.zeros(3)
        for k in range(nsat):
            spos = self.getSatellitePositionFromEphem(ephem, k)
            if np.isnan(spos[0]):
                continue
            sp0 += spos / nsat
        d0 = lin.norm(sp0)
        pos0 = sp0 / d0 * 6378137
        X0 = np.hstack((pos0, 0, 0))

        mult = np.array([10000000] * 3 + [1000, 200])
        # x,y,z,t = X0 + mult*X

        def cout(X):
            Xu = X0 + mult * X
            rpos = Xu[:3]
            dp = Xu[3]
            dv = Xu[4]

            J = 0

            for k in range(nsat):
                spos = self.getSatellitePositionFromEphem(ephem, k)
                svel = self.getSatelliteVelocityFromEphem(ephem, k)
                pr = self.getPseudorangeFromMeas(meas, k)
                vr = self.getRadialVelocityFromMeas(meas, k)
                if np.isnan(vr):
                    continue

                ab = spos - rpos
                di = lin.norm(ab)
                pr_est = di + dp
                vr_est = svel @ ab / di + dv
                err_p = pr_est - pr
                err_v = vr_est - vr

                J += err_p**2 + err_v**2

            return J

        def jac(X):
            Xu = X0 + mult * X
            rpos = Xu[:3]
            dp = Xu[3]
            dv = Xu[4]

            G = np.zeros(5)

            for k in range(nsat):
                spos = self.getSatellitePositionFromEphem(ephem, k)
                svel = self.getSatelliteVelocityFromEphem(ephem, k)
                pr = self.getPseudorangeFromMeas(meas, k)
                vr = self.getRadialVelocityFromMeas(meas, k)
                if np.isnan(vr):
                    continue

                ab = spos - rpos
                di = lin.norm(ab)
                pr_est = di + dp
                vr_est = svel @ ab / di + dv
                err_p = pr_est - pr
                err_v = vr_est - vr

                G[:3] += (
                    2 * (-svel / di + ab * (svel @ ab) / di**3) * err_v
                    - 2 * ab / di * err_p
                )
                G[3] += 2 * err_p
                G[4] += 2 * err_v

            return G * mult

        bnd = np.ones(5)

        # jac: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, trust-constr
        # bounds: L-BFGS-B, TNC, SLSQP, Powell, trust-constr
        res = minimize(
            fun=cout,
            jac=jac,
            x0=np.zeros(5),
            method=self.optim,
            # options={"maxfev": 100000, "maxiter": 100000},
            bounds=Bounds(-bnd, bnd),
        )
        if not res.success:
            return None

        Xu = X0 + mult * res.x
        pos = Xu[:3]

        return np.hstack((pos, np.zeros(3))), Xu[3], Xu[4]

    def computeFromPseudoRanges(
        self, ephem: np.array, meas: np.array
    ) -> Tuple[np.array, float]:
        """Runs a PVT algorithm that uses only peudo-range

        Args:
            ephem: Array of all ephemeris
            meas: Array of all measurements

        Returns:
            A tuple with:

            * an array of 6 scalar, 3 position (m) and 3 velocity (m/s). Velocities are set to 0
            * range bias (m)

        """
        nsat = len(meas) // 2

        # On initialise avec la position au nadir du premier satellite visible (ITRF)
        sp0 = np.zeros(3)
        for k in range(nsat):
            spos = self.getSatellitePositionFromEphem(ephem, k)
            if np.isnan(spos[0]):
                continue
            sp0 += spos / nsat
        d0 = lin.norm(sp0)
        if d0 < 1:
            pos0 = np.zeros(3)
        else:
            pos0 = sp0 / d0 * 6378137
        X0 = np.hstack((pos0, 0))

        mult = np.array([10000000] * 3 + [1000])
        # x,y,z,t = X0 + mult*X

        def cout(X):
            Xu = X0 + mult * X
            rpos = Xu[:3]
            dp = Xu[3]

            J = 0

            for k in range(nsat):
                spos = self.getSatellitePositionFromEphem(ephem, k)
                # svel = self.getSatelliteVelocityFromEphem(ephem, k)
                pr = self.getPseudorangeFromMeas(meas, k)
                vr = self.getRadialVelocityFromMeas(meas, k)
                if np.isnan(vr):
                    continue

                ab = spos - rpos
                di = lin.norm(ab)
                pr_est = di + dp
                err_p = pr_est - pr

                J += err_p**2

            return J

        def jac(X):
            Xu = X0 + mult * X
            rpos = Xu[:3]
            dp = Xu[3]

            G = np.zeros(4)

            for k in range(nsat):
                spos = self.getSatellitePositionFromEphem(ephem, k)
                # svel = self.getSatelliteVelocityFromEphem(ephem, k)
                pr = self.getPseudorangeFromMeas(meas, k)
                vr = self.getRadialVelocityFromMeas(meas, k)
                if np.isnan(vr):
                    continue

                ab = spos - rpos
                di = lin.norm(ab)
                pr_est = di + dp
                err_p = pr_est - pr

                G[:3] += -2 * ab / di * err_p
                G[3] += 2 * err_p

            return G * mult

        bnd = np.ones(4)

        # jac: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, trust-constr
        # bounds: L-BFGS-B, TNC, SLSQP, Powell, trust-constr
        res = minimize(
            fun=cout,
            jac=jac,
            x0=np.zeros(4),
            method=self.optim,
            # options={"maxfev": 100000, "maxiter": 100000},
            bounds=Bounds(-bnd, bnd),
        )
        if not res.success:
            return None

        Xu = X0 + mult * res.x
        pos = Xu[:3]
        dv = Xu[3]

        return np.hstack((pos, np.zeros(3))), dv

    def getGeocentricITRFPositionAt(self, t_calc: datetime) -> "array":
        """
        Return the geocentric ITRF position of the satellite at a given time

        Args:
            td: Time of the position

        Returns:
            An array with x, y, z (m) and vx, vy, vz (m/s)

        """
        otp = self.getOutputByName("realpos")
        realpos = otp.getGeocentricITRFPositionAt(t_calc)
        return realpos

    def getAbsoluteTime(self, t: float) -> datetime:
        """Converts a simulation time into absolute time

        Args:
            t: Simulation time (s)

        Returns:
            Absolute time in UTC

        """
        otp = self.getOutputByName("realpos")
        tsync = otp.getTsync()
        dt = timedelta(seconds=t)
        td = tsync + dt
        return td

    def update(
        self,
        t1: float,
        t2: float,
        measurements: np.array,
        ephemeris: np.array,
        realpos: np.array,
        estdop: np.array,
        estpos: np.array,
        estclkerror: np.array,
    ) -> dict:
        td = self.getAbsoluteTime(t2)

        realpos = self.getGeocentricITRFPositionAt(td)

        if np.max(np.abs(measurements)) < 1 or not self.algo:
            pos = np.zeros(6)
            dp = 0
            dv = 0
            sx = sy = sz = sp = sv = 0
        elif self.algo == "ranging":
            pos, dp = self.computeFromPseudoRanges(ephemeris, measurements)
            dv = 0
            sx, sy, sz, sp, sv = self.getDOP(ephemeris, realpos)
        elif self.algo == "doppler":
            pos, dv = self.computeFromRadialVelocities(ephemeris, measurements)
            dp = 0
            sx, sy, sz, sp, sv = self.getDOP(ephemeris, realpos)
        elif self.algo == "doppler-ranging":
            pos, dp, dv = self.computeFromPRandVR(ephemeris, measurements)
            sx, sy, sz, sp, sv = self.getDOP(ephemeris, realpos)

        estdop = np.array([sx, sy, sz, sp, sv])

        outputs = {}

        outputs["realpos"] = realpos
        outputs["estpos"] = pos
        outputs["estclkerror"] = np.array([dp, dv])
        outputs["estdop"] = estdop

        return outputs
