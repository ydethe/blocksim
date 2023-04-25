import scipy.linalg as lin

import numpy as np

from ..control.Sensors import ASensors
from ..core.Node import AWGNOutput

from ..constants import mu, omega
from ..utils import FloatArr, itrf_to_azeld


class GNSSTracker(ASensors):
    """Tracker of satellites. Provides a set of measurements (ranging and radial velocity)

    The inputs of the computer are **state** and **ueposition**
    The output of the computer are **measurement** and **obscoord**

    The **state** vector contains, for the satellite k:

    * satellite's position (m) in ITRF in state[6*k:6*k+3]
    * satellite's velocity (m/s) in ITRF in state[6*k+3:6*k+6]

    The **ueposition** vector contains:

    * UE's position (m) in ITRF in ueposition[:3]
    * UE's velocity (m/s) in ITRF in ueposition[3:]

    The **measurement** vector contains, for the satellite k:

    * satellite's position (m) in ITRF in meas[8*k:8*k+3]
    * satellite's velocity (m/s) in ITRF in meas[8*k+3:8*k+6]
    * pseudorange for the satellite (m) in meas[8*k+6]
    * radial velocity for the satellite (m/s) in meas[8*k+7]

    The **obscoord** vector contains, for the satellite k:

    * Azimut (rad) in obscoord[6*k]
    * Elevation (rad) in obscoord[6*k+1]
    * Distance (m) in obscoord[6*k+2]
    * Radial velocity (m/s) in obscoord[6*k+3]
    * Radial acceleration (m/s^2) in obscoord[6*k+4]
    * Radial jerk (m/s^3) in obscoord[6*k+5]

    * satellite's position (m) in ITRF in meas[8*k:8*k+3]
    * satellite's velocity (m/s) in ITRF in meas[8*k+3:8*k+6]
    * pseudorange for the satellite (m) in meas[8*k+6]
    * radial velocity for the satellite (m/s) in meas[8*k+7]

    The attributes are to be defined by the user :

    Attributes:
        mean: Mean of the gaussian noise. Dimension (n,1)
        cov: Covariance of the gaussian noise. Dimension (n,n)
        cho: Cholesky decomposition of cov, computed after a first call to *updateAllOutput*.
             Dimension (n,n)
        elev_mask: Elevation mask to determine if a satellite is visible (rad)
        dp: Systematic error on the ranging measurement (m)
        dv: Systematic error on the radial velocity measurement (m/s)

    Args:
        name: Name of the element
        nsat: Number of satellites followed by the tracker

    """

    __slots__ = []

    def __init__(self, name: str, nsat: int):
        nom_meas = ["pr", "vr"]
        nom_ephem = ["px", "py", "pz", "vx", "vy", "vz"]
        nom_obscoord = ["azim", "elev", "dist", "vrad", "arad", "jrad"]

        cpt_snames = []
        eph_snames = []
        obs_snames = []
        for k in range(nsat):
            cpt_snames.extend(["%s%i" % (n, k) for n in nom_meas])
            eph_snames.extend(["%s%i" % (n, k) for n in nom_ephem])
            obs_snames.extend(["%s%i" % (n, k) for n in nom_obscoord])

        ASensors.__init__(
            self,
            name=name,
            shape_state=(6 * nsat,),
            snames=cpt_snames,
            dtype=np.float64,
        )
        self.defineInput("ueposition", shape=(6,), dtype=np.float64)
        otp = AWGNOutput(name="ephemeris", snames=eph_snames, dtype=np.float64)
        otp.setInitialState(np.zeros(otp.getDataShape(), dtype=otp.getDataType()))
        self.addOutput(otp)
        self.defineOutput(name="vissat", snames=["n"], dtype=np.int64)
        self.defineOutput(name="obscoord", snames=obs_snames, dtype=np.float64)
        self.createParameter("elev_mask", value=0)
        self.createParameter("dp", value=0)
        self.createParameter("dv", value=0)

        self.setMean(np.zeros(2 * nsat), oname="measurement")
        self.setCovariance(np.zeros((2 * nsat, 2 * nsat)), oname="measurement")

        self.setMean(np.zeros(6 * nsat), oname="ephemeris")
        self.setCovariance(np.zeros((6 * nsat, 6 * nsat)), oname="ephemeris")

    def update(
        self,
        t1: float,
        t2: float,
        measurement: FloatArr,
        ephemeris: FloatArr,
        ueposition: FloatArr,
        state: FloatArr,
        vissat: FloatArr,
        obscoord: FloatArr,
    ) -> dict:
        nsat = len(state) // 6

        meas = np.empty(2 * nsat)
        ephemeris = np.empty(6 * nsat)
        obscoord = np.empty(6 * nsat)

        vissat = np.array([0])
        for k in range(nsat):
            spv = state[6 * k : 6 * k + 6]
            p_s = spv[:3]
            v_s = spv[3:]
            p_t = ueposition[:3]

            # Calcul elevation
            azim, elev, dist, _, _, vrad = itrf_to_azeld(ueposition, spv)

            # Quelques variables auxiliaires
            dst = p_s[:3] - p_t[:3]
            u = dst / dist
            r_s = lin.norm(p_s)
            om = np.array([0, 0, omega])

            # Calcul des accélération et jerk du satellite en ECEF
            a_s = -mu / r_s**3 * p_s - np.cross(om, np.cross(om, p_s)) - 2 * np.cross(om, v_s)
            j_s = (
                -mu / r_s**3 * v_s
                + 3 * mu / r_s**5 * (p_s @ v_s) * p_s
                - np.cross(om, np.cross(om, v_s))
                - 2 * np.cross(om, a_s)
            )

            # Calcul arad et jrad
            arad = -(vrad**2) / dist + v_s @ v_s / dist + a_s @ u
            jrad = -3 * arad * vrad / dist + 3 * v_s @ a_s / dist + j_s @ u

            obscoord[6 * k] = azim
            obscoord[6 * k + 1] = elev
            obscoord[6 * k + 2] = dist
            obscoord[6 * k + 3] = vrad
            obscoord[6 * k + 4] = arad
            obscoord[6 * k + 5] = jrad

            # Calcul pseudo-distance
            pr = dist + self.dp

            # Calcul vitesse radiale
            pvr = vrad + self.dv

            # Validation avec le masque d'elevation
            if elev > self.elev_mask:
                ephemeris[6 * k : 6 * k + 6] = state[6 * k : 6 * k + 6]
                meas[2 * k] = pr
                meas[2 * k + 1] = pvr
                vissat += 1
            else:
                ephemeris[6 * k : 6 * k + 6] = np.nan
                meas[2 * k : 2 * k + 2] = np.nan

        outputs = {}
        outputs["measurement"] = meas
        outputs["ephemeris"] = ephemeris
        outputs["obscoord"] = obscoord
        outputs["vissat"] = vissat

        return outputs
