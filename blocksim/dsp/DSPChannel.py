from typing import Any

from nptyping import NDArray, Shape
import numpy as np
from numpy import exp, pi, sqrt, cos, tan

from ..core.Node import AComputer, AWGNOutput

from .DelayLine import FiniteDelayLine
from ..constants import kb, c, Req
from ..utils import itrf_to_azeld, itrf_to_geodetic
from .klobuchar import klobuchar

__all__ = ["DSPChannel"]


class DSPChannel(AComputer):
    """Adds gaussian noise to the inputs
    If dtype is complex, the generated noise will be complex.

    The inputs of the computer are **tx_pos**, **rx_pos** and **tx_sig**
    The outputs of the computer are **rx_sig** and **info**

    The **tx_pos** and **rx_pos** vectors represent a 3D position (m) and 3D velocity (m/s) in ITRF
    The output **info** stores:

    * cn0 (lin)
    * snr (lin)
    * dist (m)
    * vrad (m/s)
    * azim (deg)
    * elev (deg)
    * losses (lin)
    * delay (s)

    **tx_sig** is the RX signal

    Attributes:
        wavelength: value passed to __init__
        antenna_gain: value passed to __init__
        antenna_temp: value passed to __init__
        bandwidth: value passed to __init__
        noise_factor: value passed to __init__
        alpha: value passed to __init__
        beta: value passed to __init__
        nodop: value passed to __init__
        noatm: value passed to __init__

    Args:
        name: Name of the spectrum
        wavelength: Wavelength of the carrier (m)
        antenna_gain: Gain of the RX antenna (dB)
        antenna_temp: Temperature of the RX antenna (K)
        bandwidth: Bandwidth of the receiver (Hz)
        noise_factor: Noise factor of the receiver (dB)
        alpha: Alpha parameters for Klobuchar
        beta: Alpha parameters for Klobuchar
        num_src: Number of radiating elements
        nodop: Remove distance (delay) & Doppler effects
        noatm: Remove atmospheric effects

    """

    __slots__ = ["__gain_coef", "__delay_lines"]

    def __init__(
        self,
        name: str,
        wavelength: float,
        antenna_gain: float,
        antenna_temp: float,
        bandwidth: float,
        noise_factor: float,
        alpha: NDArray[Any, Any],
        beta: NDArray[Any, Any],
        num_src: int = 1,
        nodop: bool = False,
        noatm: bool = False,
        dtype=np.complex128,
    ):
        AComputer.__init__(self, name=name)

        self.defineInput("txpos", shape=6 * num_src, dtype=np.float64)
        self.defineInput("rxpos", shape=6, dtype=np.float64)
        self.defineInput("txsig", shape=num_src, dtype=dtype)
        otp = AWGNOutput(name="rxsig", snames=["y"], dtype=dtype)

        otp.setInitialState(np.zeros(1, dtype=otp.getDataType()))
        self.addOutput(otp)
        self.defineOutput(
            name="info",
            snames=["cn0", "snr", "dist", "vrad", "azim", "elev", "losses", "delay"],
            dtype=np.float64,
        )

        self.createParameter(name="wavelength", value=wavelength, read_only=True)
        self.createParameter(name="antenna_gain", value=antenna_gain, read_only=True)
        self.createParameter(name="antenna_temp", value=antenna_temp, read_only=True)
        self.createParameter(name="bandwidth", value=bandwidth, read_only=True)
        self.createParameter(name="noise_factor", value=noise_factor, read_only=True)
        self.createParameter(name="alpha", value=alpha, read_only=True)
        self.createParameter(name="beta", value=beta, read_only=True)
        self.createParameter(name="num_src", value=num_src, read_only=True)
        self.createParameter(name="nodop", value=nodop)
        self.createParameter(name="noatm", value=noatm)

        T0 = 290.0
        self.__gain_coef = wavelength * 10 ** (antenna_gain / 20.0) / (4 * pi)
        noise_pow = (
            kb * bandwidth * ((10 ** (noise_factor / 10) - 1) * T0 + antenna_temp)
        )
        cov = np.eye(1) * noise_pow
        mean = np.zeros(1)

        self.setMean(mean)
        self.setCovariance(cov)

        self.__delay_lines = []
        for k in range(num_src):
            dl = FiniteDelayLine(size=128, dtype=np.complex128)
            self.__delay_lines.append(dl)

    def resetCallback(self, t0: float):
        super().resetCallback(t0)
        for dl in self.__delay_lines:
            dl.reset()

    def setCovariance(self, cov: NDArray[Any, Any]):
        """Sets the covariance matrix of the gaussian distribution

        Args:
            cov: Covariance matrix

        """
        otp = self.getOutputByName("rxsig")
        n = otp.getDataShape()[0]
        if cov.shape != (n, n):
            raise ValueError(cov.shape, (n, n))
        otp.cov = cov

    def setMean(self, mean: NDArray[Any, Any]):
        """Sets the mean vector of the gaussian distribution

        Args:
            mean: Mean vector matrix

        """
        otp = self.getOutputByName("rxsig")
        n = otp.getDataShape()[0]
        if mean.shape[0] != n:
            raise ValueError(mean.shape[0], n)
        otp.mean = mean

    def getCovariance(self) -> NDArray[Any, Any]:
        """Returns the covariance matrix of the gaussian distribution

        Returns:
            Covariance matrix

        """
        otp = self.getOutputByName("rxsig")
        return otp.cov

    def getMean(self) -> NDArray[Any, Any]:
        """Returns the mean vector of the gaussian distribution

        Returns:
            Mean vector matrix

        """
        otp = self.getOutputByName("rxsig")
        return otp.mean

    def atmosphericModel(self, tx_pos: NDArray[Any, Any], rx_pos: NDArray[Any, Any]):  # type: ignore
        """Computes the atmospheric contribution

        Args:
          tx_pos: Position of the emitter (ITRF, m, m/s)
          rx_pos: Position of the receiver (ITRF, m, m/s)

        Returns:
            A tuple containing:

            * dist: distance between RX and TX (m)
            * vrad: radial velocity between  RX and TX (m/s)
            * azim: azimut angle (deg)
            * elev: elevation angle (deg)
            * L_atm: atmospheric attenuation (lin)
            * dt_atm: atmospheric delay (s)

        """
        azim, elev, dist, vrad, _, _ = itrf_to_azeld(rx_pos, tx_pos)
        lon, lat, h = itrf_to_geodetic(rx_pos)

        # Saastamoinen troposhperic model
        # https://gnss-sdr.org/docs/sp-blocks/pvt/#saastamoinen
        # https://www.cv.nrao.edu/~demerson/ionosphere/atten/atten.htm
        # https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-8-200304-S!!PDF-E.pdf
        z = pi / 2 - elev * pi / 180
        h_rel = 50.0
        p = 1013.15 * (1 - 2.2557e-5 * h) ** 5.2568
        T = 15 - 6.5e-3 * h + 273.15
        e = 6.108 * exp((17.15 * T - 4684) / (T - 38.45)) * h_rel / 100
        dt_tropo = 1 / c * 0.002277 / cos(z) * (p + e * (1255 / T + 0.05) - tan(z) ** 2)
        L_tropo = 1.0

        # Ionosphere model
        dt_iono = klobuchar(
            phi=lat,
            lbd=lon,
            elev=elev * pi / 180,
            azimuth=azim * pi / 180,
            tow=0,
            alpha=self.alpha,
            beta=self.beta,
        )
        h_iono = 100e3
        inv_sq_f = self.wavelength**2 / c**2
        L_iono_dB = 2.2e15 * inv_sq_f / sqrt(cos(z) ** 2 + 2 * h_iono / Req)
        L_iono = 10 ** (L_iono_dB / 10)

        # Combined effects
        dt_atm = dt_iono + dt_tropo
        L_atm = L_iono * L_tropo

        return dist, vrad, azim, elev, L_atm, dt_atm

    def update(
        self,
        t1: float,
        t2: float,
        txpos: NDArray[Any, Any],  # type: ignore
        rxpos: NDArray[Any, Any],  # type: ignore
        txsig: NDArray[Any, Any],  # type: ignore
        rxsig: NDArray[Any, Any],  # type: ignore
        info: NDArray[Any, Any],  # type: ignore
    ) -> dict:
        rxsig = np.empty(self.num_src, dtype=np.complex128)
        delays = np.empty(self.num_src, dtype=np.float64)
        for kelem in range(self.num_src):
            txpos_k = txpos[6 * kelem : 6 * kelem + 6]
            if self.noatm:
                azim, elev, d, vrad, _, _ = itrf_to_azeld(rxpos, txpos_k)
                L_atm = 1
                dt_atm = 0
            else:
                d, vrad, azim, elev, L_atm, dt_atm = self.atmosphericModel(
                    txpos_k, rxpos
                )

            if self.nodop:
                phi_d0 = 0.0
                vrad = 0.0
                d = 1.0
            else:
                phi_d0 = -2 * pi * (d + c * dt_atm) / self.wavelength

            delay = d / c + dt_atm
            delays[kelem] = delay

            psig = txsig[kelem] * self.__gain_coef / sqrt(L_atm) / d * exp(1j * phi_d0)

            dl = self.__delay_lines[kelem]
            dl.addSample(t2, psig)
            rx_dl_sig = dl.getDelayedSample(delay)
            rxsig[kelem] = rx_dl_sig

            if kelem == 0:
                C = np.abs(rx_dl_sig) ** 2
                N = self.getCovariance()[0, 0]
                SNR = C / N
                CN0 = SNR * self.bandwidth
                # r = lin.norm(txpos_k)
                # Rh = lin.norm(rxpos)
                # ctheta = (r**2 + d**2 - Rh**2) / (2 * d * r)
                # ctheta = np.clip(ctheta, a_min=-1, a_max=1)
                # theta = arccos(ctheta)
                info = np.array([CN0, SNR, d, vrad, azim, elev, L_atm, dt_atm])

        outputs = {}
        outputs["rxsig"] = np.array([np.sum(rxsig)], dtype=np.complex128)
        outputs["info"] = info

        return outputs
