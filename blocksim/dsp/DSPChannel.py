import numpy as np
from numpy import exp, pi, sqrt, cos, sin, tan
from scipy import linalg as lin

from blocksim.core.Frame import Frame
from blocksim.core.Node import AComputer, AWGNOutput
from blocksim.dsp.DSPSignal import DSPSignal

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

    Args:
      name
        Name of the spectrum
      wavelength (m)
        Wavelength of the carrier
      antenna_gain (dB)
        Gain of the antenna
      antenna_temp (K)
        Temperature of the antenna
      bandwidth (Hz)
        Bandwidth of the receiver
      noise_factor
        Noise factor of the receiver (dB)
      alpha
        Alpha parameters for Klobuchar
      beta
        Alpha parameters for Klobuchar
      nodop
        Remove distance (delay) & Doppler effects

    """

    __slots__ = ["__gain_coef", "__delay_line"]

    def __init__(
        self,
        name: str,
        wavelength: float,
        antenna_gain: float,
        antenna_temp: float,
        bandwidth: float,
        noise_factor: float,
        alpha: np.array,
        beta: np.array,
        nodop: bool = False,
        noatm: bool = False,
        dtype=np.complex128,
    ):
        AComputer.__init__(self, name=name)

        self.defineInput("txpos", shape=6, dtype=np.float64)
        self.defineInput("rxpos", shape=6, dtype=np.float64)
        self.defineInput("txsig", shape=1, dtype=dtype)
        otp = AWGNOutput(name="rxsig", snames=["y"], dtype=dtype)

        otp_size = 1

        otp.setInitialState(np.zeros(otp_size, dtype=otp.getDataType()))
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
        self.createParameter(name="nodop", value=nodop)
        self.createParameter(name="noatm", value=noatm)

        T0 = 290.0
        self.__gain_coef = wavelength * 10 ** (antenna_gain / 20.0) / (4 * pi)
        noise_pow = (
            kb * bandwidth * ((10 ** (noise_factor / 10) - 1) * T0 + antenna_temp)
        )
        cov = np.eye(otp_size) * noise_pow
        mean = np.zeros(otp_size)

        self.setMean(mean)
        self.setCovariance(cov)

        self.__delay_line = FiniteDelayLine(size=128, dtype=np.complex128)

    def setCovariance(self, cov: np.array):
        """Sets the covariance matrix of the gaussian distribution

        Args:
          cov
            Covariance matrix

        """
        otp = self.getOutputByName("rxsig")
        n = otp.getDataShape()[0]
        if cov.shape != (n, n):
            raise ValueError(cov.shape, (n, n))
        otp.cov = cov

    def setMean(self, mean: np.array):
        """Sets the mean vector of the gaussian distribution

        Args:
          mean
            Mean vector matrix

        """
        otp = self.getOutputByName("rxsig")
        n = otp.getDataShape()[0]
        if mean.shape[0] != n:
            raise ValueError(mean.shape[0], n)
        otp.mean = mean

    def getCovariance(self) -> np.array:
        """Returns the covariance matrix of the gaussian distribution

        Returns:
          Covariance matrix

        """
        otp = self.getOutputByName("rxsig")
        return otp.cov

    def getMean(self) -> np.array:
        """Returns the mean vector of the gaussian distribution

        Returns:
          Mean vector matrix

        """
        otp = self.getOutputByName("rxsig")
        return otp.mean

    def atmosphericModel(self, tx_pos: np.array, rx_pos: np.array):
        azim, elev, dist, vrad, _, _ = itrf_to_azeld(rx_pos, tx_pos)
        z = pi / 2 - elev * pi / 180
        lon, lat, h = itrf_to_geodetic(rx_pos)

        # Saastamoinen troposhperic model
        # https://gnss-sdr.org/docs/sp-blocks/pvt/#saastamoinen
        # https://www.cv.nrao.edu/~demerson/ionosphere/atten/atten.htm
        # https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-8-200304-S!!PDF-E.pdf
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

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        txpos: np.array,
        rxpos: np.array,
        txsig: np.array,
        rxsig: np.array,
        info: np.array,
    ) -> dict:
        if self.noatm:
            azim, elev, d, vrad, _, _ = itrf_to_azeld(rxpos, txpos)
            L_atm = 1
            dt_atm = 0
        else:
            d, vrad, azim, elev, L_atm, dt_atm = self.atmosphericModel(txpos, rxpos)

        if self.nodop:
            phi_d0 = 0.0
            vrad = 0.0
            d = 1.0
        else:
            phi_d0 = -2 * pi * (d + c * dt_atm) / self.wavelength

        delay = d / c + dt_atm

        rxsig = txsig[0] * self.__gain_coef / sqrt(L_atm) / d * exp(1j * phi_d0)

        self.__delay_line.addSample(t2, rxsig)
        rx_dl_sig = self.__delay_line.getDelayedSample(delay)

        C = np.abs(rx_dl_sig) ** 2
        N = self.getCovariance()[0, 0]
        SNR = C / N
        CN0 = SNR * self.bandwidth

        outputs = {}
        outputs["rxsig"] = np.array([rx_dl_sig])
        outputs["info"] = np.array([CN0, SNR, d, vrad, azim, elev, L_atm, dt_atm])

        return outputs
