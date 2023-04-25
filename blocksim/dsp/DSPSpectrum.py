from typing import TYPE_CHECKING
import numpy as np
from numpy import exp, pi
from numpy.fft import ifft, fftshift
from scipy.signal import get_window

from ..utils import FloatArr

from .DSPLine import DSPRectilinearLine
from .DSPMap import DSPRectilinearMap
from ..core.CircularBuffer import CircularBuffer
from ..core.Node import AComputer
from ..loggers.Logger import Logger

if TYPE_CHECKING:
    from .DSPSignal import DSPSignal
else:
    DSPSignal = "blocksim.dsp.DSPSignal.DSPSignal"


__all__ = ["DSPSpectrum", "RecursiveSpectrumEstimator"]


class DSPSpectrum(DSPRectilinearLine):
    """Spectrum of a signal

    Args:
        name: Name of the spectrum
        samplingStart: First frequency of the sample of the spectrum (Hz)
        samplingPeriod: Frequency spacing of the spectrum (Hz)
        y_serie: Complex samples of the spectrum
        default_transform: Function to apply to the samples before plotting.
          Shall be vectorized

    """

    def __init__(
        self,
        name: str,
        samplingStart: float = None,
        samplingPeriod: float = None,
        y_serie: FloatArr = None,
        default_transform=DSPRectilinearLine.to_db_lim(-80),
        name_of_x_var: str = "Frequency",
        unit_of_x_var: str = "Hz",
        unit_of_y_var: str = "dB",
        name_of_y_var: str = "",
    ):
        DSPRectilinearLine.__init__(
            self,
            name=name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=y_serie,
            default_transform=default_transform,
            name_of_x_var=name_of_x_var,
            unit_of_x_var=unit_of_x_var,
            unit_of_y_var=unit_of_y_var,
            name_of_y_var=name_of_y_var,
        )

    @property
    def energy(self) -> float:
        return np.real(self.y_serie @ self.y_serie.conj() * len(self))

    def ifft(self, win: str = "ones") -> DSPSignal:
        """Applies the inverse discrete Fourier transform

        Args:
            win: The window to be applied. See `blocksim.dsp.get_window`

        Returns:
          The resulting DSPSignal

        """
        from .DSPSignal import DSPSignal

        n = len(self)
        w = get_window(win, n)
        y = ifft(fftshift(self.y_serie * w) * n)
        df = self.samplingPeriod

        return DSPSignal(
            name="Temp. %s" % self.name,
            samplingStart=0,
            samplingPeriod=1 / n / df,
            y_serie=y,
        )


class RecursiveSpectrumEstimator(AComputer):
    r"""Spectrum estimator.

    The input of the element is **measurement**

    Args:
        name: Name of the system
        dt: The sampling period of the inpnut  signal
        nfft: Number of samples used to compute the spectrum

    """

    __slots__ = ["__x_buf", "__vec"]

    def __init__(
        self,
        name: str,
        dt: float,
        nfft: int,
    ):
        AComputer.__init__(
            self,
            name=name,
        )
        self.createParameter("dt", dt)
        self.createParameter("nfft", nfft)

        self.__x_buf = CircularBuffer(size=nfft, dtype=np.complex128)

        self.defineInput("measurement", shape=1, dtype=np.complex128)
        self.defineOutput(
            name="spectrum",
            snames=[f"y_est_{i}" for i in range(nfft)],
            dtype=np.complex128,
        )

    def getSpectrogram(self, log: Logger) -> DSPRectilinearMap:
        """Gets the map from the Logger after simulation

        Args:
            log: The Logger after simulation

        Returns:
            The map

        """
        t_sim = log.getValue("t")
        fs = 1 / (t_sim[1] - t_sim[0])

        img = np.empty((self.nfft, len(t_sim)), dtype=np.complex128)
        otp = self.getOutputByName("spectrum")
        ns = otp.getScalarNames()
        for k in range(self.nfft):
            vname = "%s_%s_%s" % (self.getName(), otp.getName(), ns[k])
            x = log.getValue(vname)
            img[k, :] = x

        spg = DSPRectilinearMap(
            name="map",
            samplingXStart=-self.nfft / fs / 2 + t_sim[0],
            samplingXPeriod=1 / fs,
            samplingYStart=-fs / 2,
            samplingYPeriod=fs / self.nfft,
            img=fftshift(img, axes=0),
            default_transform=np.abs,
        )
        spg.name_of_x_var = "Time"
        spg.unit_of_x_var = "s"
        spg.name_of_y_var = "Frequency"
        spg.unit_of_y_var = "Hz"

        return spg

    def resetCallback(self, t0: float):
        super().resetCallback(t0)
        self.__x_buf.reset()
        n = np.arange(self.nfft)
        self.__vec = exp(2 * pi * 1j * n / self.nfft)

    def update(
        self,
        t1: float,
        t2: float,
        measurement: FloatArr,
        spectrum: FloatArr,
    ) -> dict:
        self.__x_buf.append(measurement[0])
        prev_meas = self.__x_buf[0]
        spectrum = self.__vec * (spectrum - prev_meas + measurement[0])
        # print(prev_meas , measurement[0])

        outputs = {}
        outputs["spectrum"] = spectrum

        return outputs
