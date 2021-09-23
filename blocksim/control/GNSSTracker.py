import scipy.linalg as lin
import numpy as np
from numpy import pi
from blocksim.control.Sensors import ASensors
from ..core.Node import AWGNOutput

from .. import logger
from ..constants import c as clum
from ..utils import pdot, itrf_to_azeld


class GNSSTracker(ASensors):
    """Tracker of satellites. Provides a set of measurements (ranging and radial velocity)

    The inputs of the computer are **state** and **ueposition**
    The output of the computer is **measurement**

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

    The following parameters are to be defined by the user :

    * mean : Mean of the gaussian noise. Dimension (n,1)
    * cov : Covariance of the gaussian noise. Dimension (n,n)
    * cho : Cholesky decomposition of cov, computed after a first call to *updateAllOutput*. Dimension (n,n)
    * elev_mask (deg) : Elevation mask to determine if a satellite is visible
    * dp (m) : Systematic error on the ranging measurement
    * dv (m/s) : Systematic error on the radial velocity measurement

    Args:
      name
        Name of the element
      nsat
        Number of satellites flollowed by the tracker

    """

    __slots__ = []

    def __init__(self, name: str, nsat: int):
        nom_meas = ["pr", "vr"]
        nom_ephem = ["px", "py", "pz", "vx", "vy", "vz"]

        cpt_snames = []
        eph_snames = []
        for k in range(nsat):
            cpt_snames.extend(["%s%i" % (n, k) for n in nom_meas])
            eph_snames.extend(["%s%i" % (n, k) for n in nom_ephem])

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
        self.createParameter("elev_mask", value=0)
        self.createParameter("dp", value=0)
        self.createParameter("dv", value=0)

        self.setMean(np.zeros(2 * nsat), oname="measurement")
        self.setCovariance(np.zeros((2 * nsat, 2 * nsat)), oname="measurement")

        self.setMean(np.zeros(6 * nsat), oname="ephemeris")
        self.setCovariance(np.zeros((6 * nsat, 6 * nsat)), oname="ephemeris")

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        measurement: np.array,
        ephemeris: np.array,
        ueposition: np.array,
        state: np.array,
        vissat=np.array,
    ) -> dict:
        nsat = len(state) // 6
        rpos = ueposition[:3]
        rvel = ueposition[3:6]

        meas = np.empty(2 * nsat)
        ephemeris = np.empty(6 * nsat)

        vissat = np.array([0])
        for k in range(nsat):
            spv = state[6 * k : 6 * k + 6]
            spos = spv[:3]
            svel = spv[3:]

            # Calcul pseudo-distance
            ab = spos - rpos
            d = lin.norm(ab)
            pr = d + self.dp

            # Calcul vitesse radiale
            vr = (ab / d) @ (svel - rvel) + self.dv

            # Calcul elevation
            _, el, _, _, _, _ = itrf_to_azeld(ueposition, spv)

            # Validation avec le masque d'elevation
            if el > self.elev_mask:
                ephemeris[6 * k : 6 * k + 6] = state[6 * k : 6 * k + 6]
                meas[2 * k] = pr
                meas[2 * k + 1] = vr
                vissat += 1
            else:
                ephemeris[6 * k : 6 * k + 6] = np.nan
                meas[2 * k : 2 * k + 2] = np.nan

        outputs = {}
        outputs["measurement"] = meas
        outputs["ephemeris"] = ephemeris
        outputs["vissat"] = vissat

        return outputs
