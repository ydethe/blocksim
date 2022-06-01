import re
from typing import Callable, Any

from nptyping import NDArray, Shape
import numpy as np
from numpy import exp, pi, sqrt
from numpy.fft import fft, fftshift
from scipy.signal import correlate

from .. import logger
from ..control.SetPoint import ASetPoint
from . import get_window, phase_unfold
from .DSPLine import DSPLine
from .DSPSpectrum import DSPSpectrum


__all__ = ["DSPSignal"]


class DSPSignal(DSPLine, ASetPoint):
    """Temporal signal

    This element has no input
    The output name of the computer is **setpoint**

    Args:
        name: Name of the signal
        samplingStart: First date of the sample of the signal (s)
        samplingPeriod: Time spacing of the signal (s)
        y_serie: Complex samples of the signal
        projection: Axe projection. Can only be 'rectilinear'
        default_transform: Function to apply to the samples before plotting.
            Shall be vectorized

    """

    def __init__(
        self,
        name: str,
        samplingStart=None,
        samplingPeriod=None,
        y_serie: NDArray[Any, Any] = None,
        default_transform=np.real,
        projection: str = "rectilinear",
        dtype=np.complex128,
    ):
        ASetPoint.__init__(
            self,
            name=name,
            snames=[name],
            dtype=dtype,
        )
        DSPLine.__init__(
            self,
            name=name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=y_serie,
            projection=projection,
            default_transform=default_transform,
        )

        self.name_of_x_var = "Time"
        self.unit_of_x_var = "s"

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: NDArray[Any, Any],
    ) -> dict:
        otp = self.getOutputByName("setpoint")
        typ = otp.getDataType()

        outputs = {}
        samp = self.getSample(t2, complex_output=self.hasOutputComplex)
        outputs["setpoint"] = np.array([samp], dtype=typ)

        return outputs

    @classmethod
    def fromBinaryRandom(
        cls, name: str, samplingPeriod: int, size: int, seed: int = None
    ) -> "DSPSignal":
        """Creates a random signal whose samples are randomly,
        following a uniform law in the set {0,1}

        Args:
            name: Name of the signal
            samplingPeriod: Sampling period of the signal (s)
            size: Number of samples
            seed: Random seed

        """
        if not seed is None:
            np.random.seed(seed=seed)

        bs = np.random.randint(low=0, high=2, size=size)

        sig = cls(
            name=name,
            samplingPeriod=samplingPeriod,
            samplingStart=0,
            y_serie=bs,
            default_transform=np.real,
            dtype=np.int64,
        )

        sig.createParameter(name="seed", value=seed)

        return sig

    @classmethod
    def fromLinearFM(
        cls,
        name: str,
        samplingStart: float,
        samplingPeriod: float,
        tau: float,
        fstart: float,
        fend: float,
        repeats: int = 1,
    ) -> "DSPSignal":
        """Builds a signal from a linear frequency modulation (chirp)

        Args:
            name: Name of the signal
            samplingStart: First date of the sample of the signal (s)
            samplingPeriod: Sampling period of the signal (s)
            tau: Duration of the signal (s)
            fstart: Frequency at the beginning of the modulation (Hz)
            fend: Frequency at the end of the modulation (Hz)
            repeats: Number of pulses to concatenate

        Returns:
            The DSPSignal

        """
        fs = 1 / samplingPeriod
        ns = int(tau * fs)
        t = np.arange(ns) / fs
        x = exp(1j * (pi * t * (2 * fstart * tau + fend * t - fstart * t)) / tau)
        sig = cls(
            name=name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=np.repeat(x, repeats=repeats),
        )
        return sig

    @classmethod
    def fromLogger(cls, name: str, log: "Logger", param: str) -> "DSPSignal":
        """Builds a signal from a `blocksim.loggers.Logger.Logger`

        Args:
            name: Name of the signal
            log: Logger to use
            param: Name of the parameter to use to build the DSPSignal

        Returns:
            The DSPSignal

        """
        tps = log.getValue("t")
        val = log.getValue(param)
        return cls.fromTimeAndSamples(name=name, tps=tps, y_serie=val)

    @classmethod
    def fromTimeAndSamples(
        cls, name: str, tps: NDArray[Any, Any], y_serie: NDArray[Any, Any]
    ) -> "DSPSignal":
        """Builds a signal from 2 time and samples series

        Args:
            name: Name of the signal
            tps: Dates of the samples (s)
            y_serie: Complex samples

        Returns:
            The DSPSignal

        """
        t0 = tps[0]
        dt = tps[1] - t0
        err = np.abs(np.diff(tps) / dt - 1)
        ierr = np.nanargmax(err)
        max_err = err[ierr]
        if max_err > 1e-6:
            raise ValueError(
                "Time serie not equally spaced : at index %i, dt=%g and dt0=%g"
                % (ierr, tps[ierr + 1] - tps[ierr], dt)
            )

        return cls(
            name=name,
            samplingStart=t0,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=np.real,
        )

    @classmethod
    def fromPhaseLaw(
        cls, name: str, sampling_freq: float, pha: NDArray[Any, Any]
    ) -> "DSPSignal":
        """Builds a signal from a phase law

        Args:
            name: Name of the signal
            sampling_freq: Sampling frequency of the phase law (Hz)
            pha: The phase law

        Returns:
          The DSPSignal

        """
        y = np.exp(1j * pha)
        return cls(
            name=name,
            samplingStart=0,
            samplingPeriod=1 / sampling_freq,
            y_serie=y,
            default_transform=np.real,
        )

    def delay(self, tau: float) -> "DSPSignal":
        """Builds a delayed DSPSignal

        Args:
            tau: The delay (s)
              Positive if the signal starts later after delay

        Returns:
            The DSPSignal

        """
        return DSPSignal(
            name=self.name,
            samplingStart=self.samplingStart + tau,
            samplingPeriod=self.samplingPeriod,
            y_serie=self.y_serie,
            default_transform=self.default_transform,
        )

    def applyDopplerFrequency(self, fdop: float) -> "DSPSignal":
        """Builds a DSPSignal with a Doppler effet applied

        Args:
            fdop: The Doppler frequency to apply (Hz)

        Returns:
            The DSPSignal

        """
        return DSPSignal(
            name=self.name,
            samplingStart=self.samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=self.y_serie
            * np.exp(1j * 2 * np.pi * fdop * self.generateXSerie()),
            default_transform=np.real,
        )

    def applyGaussianNoise(self, pwr: float) -> "DSPSignal":
        """Builds a DSPSignal with a complex gaussian noise added

        Args:
            pwr: Power of the complex gaussian noise

        Returns:
            The DSPSignal

        """
        n = len(self)
        n = (np.random.normal(size=n) + 1j * np.random.normal(size=n)) * sqrt(pwr / 2)
        z = self.y_serie + n
        sig = DSPSignal(
            name=self.name,
            samplingStart=self.samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=z,
            default_transform=self.default_transform,
        )
        return sig

    @property
    def energy(self) -> float:
        return np.real(self.y_serie @ self.y_serie.conj())

    def fft(self, win: str = "ones") -> "DSPSpectrum":
        """Applies the discrete Fourier transform

        Args:
            win: The window to be applied. See `blocksim.dsp.get_window`

        Returns:
            The resulting `blocksim.dsp.DSPSpectrum.DSPSpectrum`

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

    def getUnfoldedPhase(self, eps: float = 1e-9) -> NDArray[Any, Any]:
        """Gets the phase law from the signal

        Args:
            eps: Threshold to test equality

        Returns:
            The unfolded phase law

        """
        return phase_unfold(self.y_serie, eps=eps)

    @property
    def hasOutputComplex(self) -> bool:
        otp = self.getOutputByName("setpoint")
        typ = otp.getDataType()

        if typ == np.complex128 or typ == np.complex64:
            comp_out = True
        else:
            comp_out = False

        return comp_out

    def correlate(self, y: "DSPSignal", win="ones") -> "DSPSignal":
        """Correlates the signal with another signal

        Args:
            y: The DSPSignal to correlate with
            win : The type of window to create. See `blocksim.dsp.get_window` for more details.

        Returns:
            The resulting DSPSignal

        """
        n = min(len(self), len(y))
        dt = min(self.samplingPeriod, y.samplingPeriod)
        if np.abs(self.generateXSerie(n - 1) - y.generateXSerie(n - 1)) < dt / 20:
            x_buf = self.y_serie
            y_buf = y.y_serie
            dt = self.samplingPeriod
        elif self.samplingPeriod > y.samplingPeriod:
            x_sync = self.resample(
                samplingStart=self.samplingStart,
                samplingPeriod=y.samplingPeriod,
                complex_output=self.hasOutputComplex,
            )
            x_buf = x_sync.y_serie
            y_buf = y.y_serie
            dt = y.samplingPeriod
        else:
            y_sync = y.resample(
                samplingStart=y.samplingStart,
                samplingPeriod=self.samplingPeriod,
                complex_output=self.hasOutputComplex,
            )
            x_buf = self.y_serie
            y_buf = y_sync.y_serie
            dt = self.samplingPeriod

        if len(x_buf) > len(y_buf):
            w = get_window(win, len(y_buf))
            z = correlate(x_buf, w * y_buf, mode="full", method="auto") / len(y_buf)
            t_start = self.samplingStart - dt * (len(y_buf) - 1)
        else:
            w = get_window(win, len(x_buf))
            z = correlate(y_buf, w * x_buf, mode="full", method="auto") / len(x_buf)
            t_start = y.samplingStart - dt * (len(x_buf) - 1)

        return DSPSignal(
            name="Correlation",
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=z,
            default_transform=self.to_db,
        )

    def autoCorrelation(self, win="ones") -> "DSPSignal":
        """Autocorrelation of the signal

        Args:
            win: The type of window to create. See `DSPSignal.correlate` for more details.

        Returns:
            The resulting DSPSignal

        """
        ac = self.correlate(self)
        return ac

    def applyFunction(self, fct: Callable) -> "DSPSignal":
        """Applies a function to all the samples

        Args:
            fct: A callable to apply to all samples

        Returns:
            The resulting DSPSignal

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
            The resulting DSPSignal

        """
        return self.applyFunction(np.conj)

    def reverse(self) -> "DSPSignal":
        """Returns the reversed signal

        Returns:
            The resulting DSPSignal

        """
        return self.applyFunction(lambda x: x[::-1])

    def __matmul__(self, y: "DSPSignal") -> "DSPSignal":
        return self.convolve(y)

    def convolve(self, y: "DSPSignal") -> "DSPSignal":
        """Returns the convolution with another DSPSignal

        Args:
            y: The DSPSignal to convolve with

        Returns:
            The resulting DSPSignal

        """
        z = y.reverse().conj()
        return self.correlate(z)

    def forceSamplingStart(self, samplingStart: float) -> "DSPSignal":
        """Moves the first sample timestamp to the spec ified value.
        All the other samples are shifted by the same value.

        Args:
            samplingStart: The new samplingStart (s)

        Returns:
            A new DSPSignal instance

        """
        res = DSPSignal(
            name=self.name,
            samplingStart=samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=self.y_serie,
            default_transform=self.default_transform,
        )
        return res

    def superheterodyneIQ(self, carrier_freq: float, bandwidth: float) -> "DSPSignal":
        """Use Single-Sideband Modulation to down convert the signal in baseband

        Args:
            carrier_freq: the frequency of the carrier (Hz)
            bandwidth: the bandwidth of the signal (Hz)

        Returns:
            The new DSPSignal

        """
        tps = self.generateXSerie()
        lo = exp(-1j * 2 * pi * carrier_freq * tps)
        y_mix = self.y_serie * lo

        q = int(np.floor(1 / self.samplingPeriod / bandwidth))
        logger.debug("Decimation : %i" % q)
        eff_bp = 1 / self.samplingPeriod / q

        from .DSPFilter import BandpassDSPFilter

        filt = BandpassDSPFilter(
            name="decim",
            f_low=0.0,
            f_high=eff_bp,
            numtaps=64,
            samplingPeriod=self.samplingPeriod,
            win="hamming",
        )
        dtf = filt.getGroupDelay()
        y_filt = filt.process(y_mix)

        # SSB : https://en.wikipedia.org/wiki/Single-sideband_modulation
        nz = int(2 * dtf / self.samplingPeriod)
        y_ssb = filt.process(y_mix[::-1])[::-1]
        y_ssb_sync = np.pad(array=y_ssb[:-nz], pad_width=(nz, 0), mode="constant")
        y_filt += y_ssb_sync

        res = DSPSignal(
            name=self.name,
            samplingStart=self.samplingStart - dtf,
            samplingPeriod=self.samplingPeriod * q,
            y_serie=y_filt[::q],
            default_transform=self.default_transform,
        )
        return res

    def integrate(
        self,
        period: float,
        n_integration: int = -1,
        offset: float = 0,
        coherent: bool = True,
    ) -> "DSPSignal":
        """Coherent integration of the signal

        Args:
            period: Duration of the period window (s)
            n_integration: Number of period windows to sum. A value of -1 means to sum everything
            offset: Time between the beginning of the signal and the first period window (s)

        Returns:
            Integrated signal

        """
        # Time of the first period window
        s_start = self.samplingStart + offset
        dt = self.samplingPeriod

        nb_samples_in_period = int(period / dt)
        n_integration_max = int(np.floor((self.samplingStop - s_start) / period))
        if n_integration == -1:
            nb_integrated_periods = n_integration_max
        else:
            nb_integrated_periods = min(n_integration_max, n_integration)

        nb_offset_samples = int(offset / dt)
        if nb_offset_samples < 0:
            n_samp = nb_integrated_periods * nb_samples_in_period
            pad_samples = np.zeros(-nb_offset_samples, dtype=self.y_serie.dtype)
            yp = np.hstack((pad_samples, self.y_serie[: n_samp + nb_offset_samples]))
        else:
            n_samp = slice(
                nb_offset_samples,
                nb_offset_samples + nb_integrated_periods * nb_samples_in_period,
            )
            yp = self.y_serie[n_samp]

        a = yp.reshape((nb_integrated_periods, nb_samples_in_period))
        if coherent:
            res = a.sum(axis=0) / nb_integrated_periods
        else:
            m = np.real(a * np.conj(a))
            res = sqrt(m.sum(axis=0) / nb_integrated_periods)

        res = DSPSignal(
            name=self.name,
            samplingStart=s_start,
            samplingPeriod=dt,
            y_serie=res,
            default_transform=self.default_transform,
        )

        return res
