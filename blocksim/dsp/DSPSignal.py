from typing import Tuple, Callable

from scipy import linalg as lin
import numpy as np
from numpy import log10, exp, pi, sqrt
from numpy.fft import fft, fftshift
from scipy.signal import get_window, resample, correlate, lfilter_zi, lfilter, firwin2

from .utils import phase_unfold
from .DSPLine import DSPLine
from ..blocks.SetPoint import ASetPoint


__all__ = ["DSPSignal"]


class DSPSignal(DSPLine, ASetPoint):
    """Spectrum of a signal

    This element has no input
    The output name of the computer is **setpoint**

    Args:
      name
        Name of the signal
      samplingStart (s)
        First date of the sample of the signal
      samplingPeriod (s)
        Time spacing of the signal
      y_serie
        Complex samples of the signal
      default_transform
        Function to apply to the samples before plotting.
        Shall be vectorized

    """

    name_of_x_var = "Time"
    unit_of_x_var = "s"

    def __init__(
        self,
        name: str,
        samplingStart=None,
        samplingPeriod=None,
        y_serie: np.array = None,
        default_transform=np.real,
    ):
        ASetPoint.__init__(
            self,
            name=name,
            snames=[name],
            dtype=np.complex128,
        )
        DSPLine.__init__(
            self,
            name=name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=y_serie,
            default_transform=default_transform,
        )

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
    ) -> dict:
        outputs = {}
        outputs["setpoint"] = np.array([self.getSample(t2)])

        return outputs

    @classmethod
    def fromLinearFM(
        cls,
        name: str,
        samplingStart: float,
        samplingPeriod: float,
        tau: float,
        fstart: float,
        fend: float,
    ) -> "DSPSignal":
        """Builds a signal from a linear frequency modulation (chirp)

        Args:
          name
            Name of the signal
          samplingStart (s)
            First date of the sample of the signal
          samplingPeriod (s)
            Sampling period of the signal
          tau (s)
            Duration of the signal
          fstart (Hz)
            Frequency at the beginning of the modulation
          fend (Hz)
            Frequency at the end of the modulation

        Returns:
          The :class:`SystemControl.dsp.DSPSignal`

        """
        fs = 1 / samplingPeriod
        ns = int(tau * fs)
        t = np.arange(ns) / fs
        x = exp(1j * (pi * t * (2 * fstart * tau + fend * t - fstart * t)) / tau)
        sig = DSPSignal(
            name=name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=x,
        )
        return sig

    @classmethod
    def fromLogger(cls, name: str, log: "Logger", param: str) -> "DSPSignal":
        """Builds a signal from a :class:`SystemControl.Logger.Logger`

        Args:
          name
            Name of the signal
          log
            :class:`SystemControl.Logger.Logger` to use
          param
            Name of the parameter to use to build the :class:`SystemControl.dsp.DSPSignal`

        Returns:
          The :class:`SystemControl.dsp.DSPSignal`

        """
        tps = log.getValue("t")
        val = log.getValue(param)
        return DSPSignal.fromTimeAndSamples(name=name, tps=tps, y_serie=val)

    @classmethod
    def fromTimeAndSamples(
        cls, name: str, tps: np.array, y_serie: np.array
    ) -> "DSPSignal":
        """Builds a signal from 2 time and samples series

        Args:
          name
            Name of the signal
          tps (s)
            Dates of the samples
          y_serie
            Complex samples

        Returns:
          The :class:`SystemControl.dsp.DSPSignal`

        """
        t0 = tps[0]
        dt = tps[1] - t0
        err = np.max(np.abs(np.diff(tps) / dt - 1))
        if err > 1e-6:
            raise ValueError("Time serie not equally spaced")

        return DSPSignal(
            name=name,
            samplingStart=t0,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=np.real,
        )

    @classmethod
    def fromPhaseLaw(cls, name: str, sampling_freq: float, pha: np.array):
        """Builds a signal from a phase law

        Args:
          name
            Name of the signal
          sampling_freq (Hz)
            Sampling frequency of the phase law
          pha
            The phase law

        Returns:
          The :class:`SystemControl.dsp.DSPSignal`

        """
        y = np.exp(1j * pha)
        return DSPSignal(
            name=name,
            samplingStart=0,
            samplingPeriod=1 / sampling_freq,
            y_serie=y,
            default_transform=np.real,
        )

    def delay(self, tau: float) -> "DSPSignal":
        """Builds a delayed :class:`SystemControl.dsp.DSPSignal`

        Args:
          tau (s)
            The delay. Positive if the signal starts later after delay.

        Returns:
          The :class:`SystemControl.dsp.DSPSignal`

        """
        return DSPSignal(
            name=self.name,
            samplingStart=self.samplingStart + tau,
            samplingPeriod=self.samplingPeriod,
            y_serie=self.y_serie,
            default_transform=self.default_transform,
        )

    def applyDopplerFrequency(self, fdop: float) -> "DSPSignal":
        """Builds a :class:`SystemControl.dsp.DSPSignal` with a Doppler effet applied

        Args:
          fdop (Hz)
            The Doppler frequency to apply

        Returns:
          The :class:`SystemControl.dsp.DSPSignal`

        """
        return DSPSignal(
            name="%s with dop" % self.name,
            samplingStart=self.samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=self.y_serie
            * np.exp(1j * 2 * np.pi * fdop * self.generateXSerie()),
            default_transform=np.real,
        )

    def applyGaussianNoise(self, pwr: float) -> "DSPSignal":
        """Builds a :class:`SystemControl.dsp.DSPSignal` with a complex gaussian noise added

        Args:
          pwr
            Power of the complex gaussian noise

        Returns:
          The :class:`SystemControl.dsp.DSPSignal`

        """
        n = len(self)
        n = (np.random.normal(size=n) + 1j * np.random.normal(size=n)) * sqrt(pwr / 2)
        return self.applyFunction(lambda x: x + n)

    def fft(self, win: str = "ones") -> "DSPSpectrum":
        """Applies the discrete Fourier transform

        Args:
          win
            The window to be applied. Should be compatible with `get_window`_.

        .. _get_window: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html

        Returns:
          The resulting :class:`SystemControl.dsp.DSPSpectrum`

        """
        from .DSPSpectrum import DSPSpectrum

        n = len(self)
        w = get_window(win, n)
        y = fftshift(fft(self.y_serie * w) / n)

        return DSPSpectrum(
            name="DSPSpectrum %s" % self.name,
            samplingStart=-0.5 / self.samplingPeriod,
            samplingPeriod=1 / self.samplingPeriod / n,
            y_serie=y,
        )

    def getUnfoldedPhase(self, eps: float = 1e-9) -> np.array:
        """Gets the phase law from the signal

        Args:
          eps
            If :math:`|z_k|<\epsilon` for a sample :math:`z_k`, then the samle if considered null

        Returns:
          The unfolded phase law

        """
        return phase_unfold(self.y_serie, eps=eps)

    def correlate(self, y: "DSPSignal") -> "DSPSignal":
        """Correlates the signal with another signal

        Args:
          y
            The :class:`SystemControl.dsp.DSPSignal` to correlate with

        Returns:
          The resulting :class:`SystemControl.dsp.DSPSignal`

        """
        n = min(len(self), len(y))
        dt = min(self.samplingPeriod, y.samplingPeriod)
        if np.abs(self.generateXSerie(n - 1) - y.generateXSerie(n - 1)) < dt / 20:
            x_buf = self.y_serie
            y_buf = y.y_serie
            dt = self.samplingPeriod
        elif self.samplingPeriod > y.samplingPeriod:
            x_sync = self.resample(
                samplingStart=self.samplingStart, samplingPeriod=y.samplingPeriod
            )
            x_buf = x_sync.y_serie
            y_buf = y.y_serie
            dt = y.samplingPeriod
        else:
            y_sync = y.resample(
                samplingStart=y.samplingStart, samplingPeriod=self.samplingPeriod
            )
            x_buf = self.y_serie
            y_buf = y_sync.y_serie
            dt = self.samplingPeriod

        if len(x_buf) > len(y_buf):
            z = correlate(x_buf, y_buf, mode="full", method="auto") / len(y_buf)
            t_start = self.samplingStart - dt * (len(y_buf) - 1)
        else:
            z = correlate(y_buf, x_buf, mode="full", method="auto") / len(x_buf)
            t_start = y.samplingStart - dt * (len(x_buf) - 1)

        return DSPSignal(
            name="Correlation",
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=z,
            default_transform=self.to_db,
        )

    def applyFunction(self, fct: Callable) -> "DSPSignal":
        """Applies a function to all the samples

        Args:
          fct
            A callable to apply to all samples

        Returns:
          The resulting :class:`SystemControl.dsp.DSPSignal`

        """
        return DSPSignal(
            name=self.name,
            samplingStart=self.samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=fct(self.y_serie),
            default_transform=self.default_transform,
        )

    def conj(self) -> "DSPSignal":
        """Returns the conjugated signal

        Returns:
          The resulting :class:`SystemControl.dsp.DSPSignal`

        """
        return self.applyFunction(np.conj)

    def reverse(self) -> "DSPSignal":
        """Returns the reversed signal

        Returns:
          The resulting :class:`SystemControl.dsp.DSPSignal`

        """
        return self.applyFunction(lambda x: x[::-1])

    def __matmul__(self, y: "DSPSignal") -> "DSPSignal":
        return self.convolve(y)

    def convolve(self, y: "DSPSignal") -> "DSPSignal":
        """Returns the convolution with another :class:`SystemControl.dsp.DSPSignal`

        Args:
          y
            The :class:`SystemControl.dsp.DSPSignal` to convolve with

        Returns:
          The resulting :class:`SystemControl.dsp.DSPSignal`

        """
        z = y.reverse().conj()
        return self.correlate(z)
