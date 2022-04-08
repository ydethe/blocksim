import os
from typing import Callable

from tqdm import tqdm
import numpy as np
from numpy import arccos, arctan, exp, pi, sqrt, cos, sin, tan, log2, arcsin, arctan2
from scipy import linalg as lin

from ..core.Node import AComputer
from .DSPSpectrogram import DSPSpectrogram

from .. import logger
from ..constants import c, Req, mu
from ..utils import cexp

__all__ = ["AntennaNetwork"]


class AntennaNetwork(AComputer):
    """Antenna network  implementation

    The inputs of the computer are **tx_pos**, **rx_pos** and **tx_sig**
    The outputs of the computer are **rx_sig**

    The **tx_pos** and **rx_pos** vectors represent a 3D position (m) and 3D velocity (m/s) in ITRF
    **tx_sig** is the RX signal

    Attributes:
        th_profile: Function that associated to an off-axis angle the distance between the antenna and the receiver
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
        * coefficients: Path to pkl file containing the coefficients for each antenna

    """

    __slots__ = ["_coeff"]

    def __init__(self, ac, dtype=np.complex128):
        AComputer.__init__(self, name=ac.name)

        self.defineInput("txpos", shape=6, dtype=np.float64)
        self.defineInput("rxpos", shape=6, dtype=np.float64)
        self.defineInput("txsig", shape=1, dtype=dtype)
        self.defineOutput("rxsig", snames=["y"], dtype=dtype)
        self.defineOutput("info", snames=["theta", "psi"], dtype=np.float64)

        self.createParameter(name="th_profile", value=ac.th_profile, read_only=True)
        self.createParameter(name="mapping", value=ac.mapping, read_only=True)
        self.createParameter(name="frequency", value=ac.freq, read_only=True)
        self.createParameter(name="hpbw", value=ac.hpbw, read_only=True)
        self.createParameter(name="wavelength", value=c / ac.freq, read_only=True)

        N = 0
        if os.path.exists(ac.coefficients):
            from pickle import load

            with open(ac.coefficients, "rb") as f:
                self._coeff = load(f)

            N = len(self._coeff)
            logger.info("Loaded '%s'" % ac.coefficients)

        self.createParameter(name="num_elem", value=N, read_only=True)

    def getCoefficients(self) -> "array":
        """Returns the coefficients applied to each antenna

        Returns:
            The coefficients

        """
        return self._coeff.copy()

    def antennaDiagram(self, n_points: int = 100) -> DSPSpectrogram:
        """Computes the antenna diagram

        Args:
            n_points: Number of points for theta and psi axis

        Returns:
            A DSPSpectrogram that represents the diagram

        """
        used_theta = np.linspace(0, pi / 2, n_points)
        used_psi = np.linspace(0, 2 * pi, n_points)
        m = np.zeros((n_points, n_points), dtype=np.complex128)
        Emax = -1
        for s, psi in enumerate(used_psi):
            for r, theta in enumerate(used_theta):
                for k in range(self.num_elem):
                    p, q = self.mapping(k)
                    d = self.th_profile(theta) - sin(theta) * (
                        cos(psi) * p + sin(psi) * q
                    )
                    m[r, s] += self._coeff[k] * cexp(d / self.wavelength)

                if np.abs(m[r, s]) > Emax:
                    Emax = np.abs(m[r, s])

        diag = DSPSpectrogram(
            name="diag",
            samplingXStart=used_psi[0],
            samplingXPeriod=used_psi[1] - used_psi[0],
            samplingYStart=used_theta[0] * 180 / pi,
            samplingYPeriod=(used_theta[1] - used_theta[0]) * 180 / pi,
            img=m / Emax,
            projection="polar",
            default_transform=DSPSpectrogram.to_db_lim(-40),
        )
        diag.name_of_x_var = ""  # Off-axis angle
        diag.unit_of_x_var = "rad"
        diag.name_of_y_var = ""  # Azimut angle
        diag.unit_of_y_var = "deg"

        return diag

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        txpos: "array",
        rxpos: "array",
        txsig: "array",
        rxsig: "array",
        info: "array",
    ) -> dict:
        u = rxpos[:3] - txpos[:3]
        d = lin.norm(u)
        u /= d
        theta = arccos(u[0])
        psi = arctan2(u[2], u[1])

        amp = 0
        d0 = self.th_profile(theta)
        for k in np.arange(self.num_elem):
            p, q = self.mapping(k)
            dk = d0 - sin(theta) * (cos(psi) * p + sin(psi) * q)
            amp += self._coeff[k] * cexp(dk / self.wavelength)

        outputs = {}
        outputs["rxsig"] = txsig * amp

        outputs["info"] = np.array([theta, psi])

        return outputs
