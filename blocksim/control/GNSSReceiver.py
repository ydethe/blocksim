from typing import List, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy import sqrt
import scipy.linalg as lin
from scipy.optimize import minimize, Bounds
from skyfield.api import Topos, load
from skyfield import framelib

from .. import logger
from ..utils import datetime_to_skyfield, skyfield_to_datetime
from ..source.Satellite import Satellite
from ..core.Node import AComputer, Output
from ..core.Frame import Frame
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

    def getGeocentricITRFPositionAt(self, t: float) -> np.array:
        """
        Return the geocentric ITRF position of the satellite at a given time

        Args:
          td
            Time of the position

        Returns:
          x, y, z (m)

        """
        dt = timedelta(seconds=t)
        td = self.__tsync + dt
        t = datetime_to_skyfield(td)

        pv = self.__sgp4_rep.at(t)
        pos, vel = pv.frame_xyz_and_velocity(framelib.itrs)

        ps = pos.m
        vs = vel.m_per_s

        return np.hstack((ps, vs))

    def resetCallback(self, frame: Frame):
        """Resets the element internal state to zero."""
        state = self.getGeocentricITRFPositionAt(frame.getStartTimeStamp())
        self.setInitialState(state)


class GNSSReceiver(AComputer):
    """GNSS receiver. Warning: before any simulation, you should call initialisePosition

    The input of the computer is **measurements**
    The outputs of the computer are **realpos**, **estpos** and **estclkerror**

    The following parameters are to be defined by the user :

    * algo : Type of algorihtm to use. Can be 'ranging' or 'doppler'

    Args:
      name
        Name of the element
      nsat
        Number of satellites flollowed by the tracker
      lon (deg)
        True longitude of the receiver
      lat (deg)
        True latitude of the receiver
      alt (m)
        True altitude of the receiver
      tsync
        Date where the calculation takes place

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

    def getSatellitePositionFromEphem(self, ephem: np.array, isat: int) -> np.array:
        return ephem[6 * isat : 6 * isat + 3]

    def getSatelliteVelocityFromEphem(self, ephem: np.array, isat: int) -> np.array:
        return ephem[6 * isat + 3 : 6 * isat + 6]

    def getPseudorangeFromMeas(self, meas: np.array, isat: int) -> np.array:
        return meas[2 * isat]

    def getRadialVelocityFromMeas(self, meas: np.array, isat: int) -> np.array:
        return meas[2 * isat + 1]

    def getDOP(self, ephem: np.array, pv_ue: np.array) -> Tuple[float]:
        """Computes the DOPs

        Args:
          pv_ue
            Ephemeris vector
          ephemeris (m)
            UE 3D position/velocity (ITRF) without velocity

        Returns:
          DOP for X axis (ITRF) (wo unit for 'ranging' algo, seconds for 'doppler' algo)
          DOP for Y axis (ITRF) (wo unit for 'ranging' algo, seconds for 'doppler' algo)
          DOP for Z axis (ITRF) (wo unit for 'ranging' algo, seconds for 'doppler' algo)
          DOP for clock error (wo unit for 'ranging' algo, seconds for 'doppler' algo)

        """
        pos = pv_ue[:3]
        nsat = len(ephem) // 6
        nval = 0
        if self.algo == "doppler-ranging":
            B = np.empty((2 * nsat, 5))
        else:
            B = np.empty((nsat, 4))

        for k in range(nsat):
            spos = self.getSatellitePositionFromEphem(ephem, k)
            svel = self.getSatelliteVelocityFromEphem(ephem, k)
            if np.isnan(spos[0]):
                continue

            R = spos - pos
            d = lin.norm(R)

            if self.algo == "doppler":
                B[nval, :3] = -svel / d + R / d ** 3 * (svel @ R)
                B[nval, 3] = 1
                nval += 1
            elif self.algo == "ranging":
                B[nval, :3] = -R / d
                B[nval, 3] = 1
                nval += 1
            elif self.algo == "doppler-ranging":
                B[nval, :3] = -R / d
                B[nval, 3] = 1
                nval += 1
                B[nval, :3] = -svel / d + R / d ** 3 * (svel @ R)
                B[nval, 4] = 1
                nval += 1

        B = B[: nval + 1, :]

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

                J += err_v ** 2

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

                G[:3] += 2 * (-svel / di + ab * (svel @ ab) / di ** 3) * err_v
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

                J += err_p ** 2 + err_v ** 2

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
                    2 * (-svel / di + ab * (svel @ ab) / di ** 3) * err_v
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

        mult = np.array([10000000] * 3 + [1000])
        # x,y,z,t = X0 + mult*X

        def cout(X):
            Xu = X0 + mult * X
            rpos = Xu[:3]
            dp = Xu[3]

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
                err_p = pr_est - pr

                J += err_p ** 2

            return J

        def jac(X):
            Xu = X0 + mult * X
            rpos = Xu[:3]
            dp = Xu[3]

            G = np.zeros(4)

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
                err_p = pr_est - pr

                G[:3] += -2 * ab / di * err_p
                G[3] += 2 * err_p

            return G * mult

        def bancroft():
            a = np.empty(nsat)
            B = np.empty((nsat, 4))
            nval = -1
            for k in range(nsat):
                spos = self.getSatellitePositionFromEphem(ephem, k)
                pr = self.getPseudorangeFromMeas(meas, k)
                if np.isnan(pr):
                    continue
                nval += 1
                a[nval] = 0.5 * (spos @ spos - pr ** 2)
                B[nval, :3] = spos
                B[nval, 3] = -pr

            a = a[: nval + 1]
            B = B[: nval + 1, :]
            e = np.ones(nval + 1)

            Bp = lin.inv(B.T @ B) @ B.T
            Bpe = Bp @ e
            Bpa = Bp @ a

            a2 = pdot(Bpe, Bpe)
            b2 = -2 + 2 * pdot(Bpe, Bpa)
            c2 = pdot(Bpa, Bpa)

            dlt = b2 ** 2 - 4 * a2 * c2
            l1 = -b2 / (2 * a2) + np.sqrt(dlt) / (2 * a2)
            l2 = -b2 / (2 * a2) - np.sqrt(dlt) / (2 * a2)

            u1 = np.dot(Bp, a + l1 * e)
            u2 = np.dot(Bp, a + l2 * e)

            v1 = np.dot(B, u1) - a - l1 * e
            v2 = np.dot(B, u2) - a - l2 * e
            j1 = np.dot(v1, v1)
            j2 = np.dot(v2, v2)

            if j1 < j2:
                pos = u1[:3]
                ct = u1[3]
            else:
                pos = u2[:3]
                ct = u2[3]

            return pos, ct

        if self.optim == "bancroft":
            pos, dv = bancroft()
        else:
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

    def compute_outputs(
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
        if np.max(np.abs(measurements)) < 1:
            pos = np.zeros(6)
            dp = 0
            dv = 0
            logger.debug("No valid measurement")
            sx = sy = sz = sp = sv = 0
        elif self.algo == "ranging":
            pos, dp = self.computeFromPseudoRanges(ephemeris, measurements)
            dv = 0
            sx, sy, sz, sp, sv = self.getDOP(ephemeris, pos)
        elif self.algo == "doppler":
            pos, dv = self.computeFromRadialVelocities(ephemeris, measurements)
            dp = 0
            sx, sy, sz, sp, sv = self.getDOP(ephemeris, pos)
        elif self.algo == "doppler-ranging":
            pos, dp, dv = self.computeFromPRandVR(ephemeris, measurements)
            sx, sy, sz, sp, sv = self.getDOP(ephemeris, pos)

        estdop = np.array([sx, sy, sz, sp, sv])

        outputs = {}

        otp = self.getOutputByName("realpos")
        outputs["realpos"] = otp.getGeocentricITRFPositionAt(t2)
        outputs["estpos"] = pos
        outputs["estclkerror"] = np.array([dp, dv])
        outputs["estdop"] = estdop

        return outputs
