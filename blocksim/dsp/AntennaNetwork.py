import numpy as np
from numpy import pi, cos, sin

from ..core.Node import AComputer
from .DSPMap import DSPPolarMap

from ..constants import c
from ..utils import FloatArr, build_local_matrix, cexp

__all__ = ["AntennaNetwork"]


class AntennaNetwork(AComputer):
    """Antenna network  implementation

    The inputs of the computer are **tx_pos**, **rx_pos** and **tx_sig**
    The outputs of the computer are **rx_sig**

    The **tx_pos** vector represents a 3D position (m) and 3D velocity (m/s) in ITRF
    **tx_sig** is the RX signal

    Attributes:
        th_profile: Function that associated to an off-axis angle the distance between the antenna
                    and the receiver
        mapping: List of antennas coordinates (m)
        frequency: Frequency of the antenna (Hz)
        hpbw: Half Power Beam Width of the antenna (rad)
        wavelength: Wavelength of the carrier (m)

    Args:
        ac: Path to a python file describing the antenna. This file shall define:

        * name: Name of the antenna (str)
        * th_profile
        * mapping
        * freq (Hz)
        * hpbw (rad)
        * coefficients: array of coefficients for each antenna

    """

    __slots__ = ["_coeff"]

    def __init__(self, ac, dtype=np.complex128):
        AComputer.__init__(self, name=ac.name)

        self._coeff = np.array(ac.coefficients, dtype=np.complex128)
        N = len(self._coeff)

        epnames = []
        for k in range(N):
            epnames.extend([f"px{k}", f"py{k}", f"pz{k}", f"vx{k}", f"vy{k}", f"vz{k}"])

        self.defineInput("txpos", shape=6, dtype=np.float64)
        self.defineInput("txsig", shape=1, dtype=dtype)
        self.defineOutput("rxsig", snames=[f"y{k}" for k in range(N)], dtype=dtype)
        self.defineOutput("elempos", snames=epnames, dtype=np.float64)

        self.createParameter(name="th_profile", value=ac.th_profile, read_only=True)
        self.createParameter(name="mapping", value=ac.mapping, read_only=True)
        self.createParameter(name="frequency", value=ac.freq, read_only=True)
        self.createParameter(name="hpbw", value=ac.hpbw, read_only=True)
        self.createParameter(name="wavelength", value=c / ac.freq, read_only=True)
        self.createParameter(name="num_elem", value=N, read_only=True)

    def getCoefficients(self) -> FloatArr:
        """Returns the coefficients applied to each antenna

        Returns:
            The coefficients

        """
        return self._coeff.copy()

    def antennaDiagram(self, n_points: int = 100) -> DSPPolarMap:
        """Computes the antenna diagram

        Args:
            n_points: Number of points for theta and psi axis

        Returns:
            A DSPPolarMap that represents the diagram

        """
        used_theta = np.linspace(0, self.hpbw, n_points)
        used_psi = np.linspace(0, 2 * pi, n_points)
        m = np.zeros((n_points, n_points), dtype=np.complex128)
        Emax = -1
        for s, psi in enumerate(used_psi):
            for r, theta in enumerate(used_theta):
                for k in range(self.num_elem):
                    p, q = self.mapping(k)
                    d = self.th_profile(theta) - sin(theta) * (cos(psi) * p + sin(psi) * q)
                    m[r, s] += self._coeff[k] * cexp(d / self.wavelength)

                if np.abs(m[r, s]) > Emax:
                    Emax = np.abs(m[r, s])

        diag = DSPPolarMap(
            name="diag",
            samplingXStart=used_psi[0],
            samplingXPeriod=used_psi[1] - used_psi[0],
            samplingYStart=used_theta[0] * 180 / pi,
            samplingYPeriod=(used_theta[1] - used_theta[0]) * 180 / pi,
            img=m / Emax,
            default_transform=DSPPolarMap.to_db_lim(-40),
        )
        diag.name_of_x_var = ""  # Off-axis angle
        diag.unit_of_x_var = "rad"
        diag.name_of_y_var = ""  # Azimut angle
        diag.unit_of_y_var = "deg"

        return diag

    def update(
        self,
        t1: float,
        t2: float,
        txpos: FloatArr,
        txsig: FloatArr,
        rxsig: FloatArr,
        elempos: FloatArr,
    ) -> dict:
        M = build_local_matrix(txpos[:3], xvec=txpos[3:])

        apos = np.zeros(3)
        for k in np.arange(self.num_elem):
            apos[:2] = self.mapping(k)
            elempos[6 * k : 6 * k + 3] = txpos[:3] + M @ apos
            elempos[6 * k + 3 : 6 * k + 6] = txpos[3:]  # TODO: Compute real velocities

        outputs = {}
        outputs["rxsig"] = txsig[0] * self._coeff
        outputs["elempos"] = elempos

        return outputs
