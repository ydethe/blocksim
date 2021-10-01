from typing import Tuple, Callable, List, Union

from scipy import linalg as lin
import numpy as np
from numpy import log10, exp, pi, sqrt
from numpy.fft import fft, fftshift
from scipy.signal import correlate, lfilter_zi, lfilter, firwin2
from .utils import get_window

from .utils import phase_unfold, zadoff_chu, shift
from .DSPLine import DSPLine
from ..control.SetPoint import ASetPoint


__all__ = ["DSPSignal"]


class DSPSignal(DSPLine, ASetPoint):
    """Temporal signal

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
            default_transform=default_transform,
        )

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
    ) -> dict:
        otp = self.getOutputByName("setpoint")
        typ = otp.getDataType()

        outputs = {}
        samp = self.getSample(t2, complex_output=self.hasOutputComplex)
        outputs["setpoint"] = np.array([samp], dtype=typ)

        return outputs

    @classmethod
    def fromRandom(
        cls, name: str, samplingPeriod: int, size: int, seed: int = None
    ) -> "DSPSignal":
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
          The :class:`blocksim.dsp.DSPSignal`

        """
        fs = 1 / samplingPeriod
        ns = int(tau * fs)
        t = np.arange(ns) / fs
        x = exp(1j * (pi * t * (2 * fstart * tau + fend * t - fstart * t)) / tau)
        sig = cls(
            name=name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=x,
        )
        return sig

    @classmethod
    def fromGoldSequence(
        cls,
        name: str,
        sv: Union[List[int], int],
        repeat=1,
        sampling_freq: float = 1.023e6,
        samplingStart: float = 0,
    ) -> "DSPSignal":
        """Builds Gold sequence

        Args:
          name
            Name of the signal
          repeat
            Number of copies of a 1023 Gold sequence
          sv
            Identifier of the SV. Can be either the PRN number (int), or the code tap selection (list of 2 int)

        Returns:
          The :class:`blocksim.dsp.DSPSignal`. All the samples are +1 or -1

        """
        SV_list = {
            1: [2, 6],
            2: [3, 7],
            3: [4, 8],
            4: [5, 9],
            5: [1, 9],
            6: [2, 10],
            7: [1, 8],
            8: [2, 9],
            9: [3, 10],
            10: [2, 3],
            11: [3, 4],
            12: [5, 6],
            13: [6, 7],
            14: [7, 8],
            15: [8, 9],
            16: [9, 10],
            17: [1, 4],
            18: [2, 5],
            19: [3, 6],
            20: [4, 7],
            21: [5, 8],
            22: [6, 9],
            23: [1, 3],
            24: [4, 6],
            25: [5, 7],
            26: [6, 8],
            27: [7, 9],
            28: [8, 10],
            29: [1, 6],
            30: [2, 7],
            31: [3, 8],
            32: [4, 9],
        }

        if not hasattr(sv, "__iter__"):
            sv = SV_list[sv]

        # init registers
        G1 = [1 for _ in range(10)]
        G2 = [1 for _ in range(10)]

        ca = []
        for _ in range(1023):
            g1 = shift(G1, [3, 10], [10])  # feedback 3,10, output 10
            g2 = shift(
                G2, [2, 3, 6, 8, 9, 10], sv
            )  # feedback 2,3,6,8,9,10, output sv for sat
            ca.append((g1 + g2) % 2)

        seq = -1 + 2 * np.array(ca * repeat, dtype=np.int8)
        sig = cls(
            name=name, samplingStart=0, samplingPeriod=1 / sampling_freq, y_serie=seq
        )
        return sig

    @classmethod
    def fromZadoffChu(
        cls,
        name: str,
        n_zc: int,
        u: int,
        sampling_freq: float,
        samplingStart: float = 0,
    ) -> "DSPSignal":
        """Builds Zadoff-Chu sequence

        Args:
          name
            Name of the signal
          n_zc
            Length of the Zadoff-Chu sequence
          u
            Index of the Zadoff-Chu sequence

        Returns:
          The :class:`blocksim.dsp.DSPSignal`

        """
        seq = zadoff_chu(u, n_zc)
        sig = cls(
            name=name, samplingStart=0, samplingPeriod=1 / sampling_freq, y_serie=seq
        )
        return sig

    @classmethod
    def fromLogger(cls, name: str, log: "Logger", param: str) -> "DSPSignal":
        """Builds a signal from a :class:`blocksim.Logger.Logger`

        Args:
          name
            Name of the signal
          log
            :class:`blocksim.Logger.Logger` to use
          param
            Name of the parameter to use to build the :class:`blocksim.dsp.DSPSignal`

        Returns:
          The :class:`blocksim.dsp.DSPSignal`

        """
        tps = log.getValue("t")
        val = log.getValue(param)
        return cls.fromTimeAndSamples(name=name, tps=tps, y_serie=val)

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
          The :class:`blocksim.dsp.DSPSignal`

        """
        t0 = tps[0]
        dt = tps[1] - t0
        err = np.max(np.abs(np.diff(tps) / dt - 1))
        if err > 1e-6:
            raise ValueError("Time serie not equally spaced")

        return cls(
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
          The :class:`blocksim.dsp.DSPSignal`

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
        """Builds a delayed :class:`blocksim.dsp.DSPSignal`

        Args:
          tau (s)
            The delay. Positive if the signal starts later after delay.

        Returns:
          The :class:`blocksim.dsp.DSPSignal`

        """
        return DSPSignal(
            name=self.name,
            samplingStart=self.samplingStart + tau,
            samplingPeriod=self.samplingPeriod,
            y_serie=self.y_serie,
            default_transform=self.default_transform,
        )

    def applyDopplerFrequency(self, fdop: float) -> "DSPSignal":
        """Builds a :class:`blocksim.dsp.DSPSignal` with a Doppler effet applied

        Args:
          fdop (Hz)
            The Doppler frequency to apply

        Returns:
          The :class:`blocksim.dsp.DSPSignal`

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
        """Builds a :class:`blocksim.dsp.DSPSignal` with a complex gaussian noise added

        Args:
          pwr
            Power of the complex gaussian noise

        Returns:
          The :class:`blocksim.dsp.DSPSignal`

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

    def fft(self, win: str = "ones") -> "DSPSpectrum":
        """Applies the discrete Fourier transform

        Args:
          win
            The window to be applied. Should be compatible with `get_window`_.

        .. _get_window: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html

        Returns:
          The resulting :class:`blocksim.dsp.DSPSpectrum`

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
          y
            The :class:`blocksim.dsp.DSPSignal` to correlate with
          win : string, float, or tuple
            The type of window to create. See below for more details.

        Returns:
          The resulting :class:`blocksim.dsp.DSPSignal`

        Notes:
          Window types:
            - boxcar
            - triang
            - blackman
            - hamming
            - hann
            - bartlett
            - flattop
            - parzen
            - bohman
            - blackmanharris
            - nuttall
            - barthann
            - cosine
            - exponential
            - tukey
            - taylor
            - kaiser (needs beta)
            - gaussian (needs standard deviation)
            - general_cosine (needs weighting coefficients)
            - general_gaussian (needs power, width)
            - general_hamming (needs window coefficient)
            - dpss (needs normalized half-bandwidth)
            - chebwin (needs attenuation)

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

    def autoCorrelation(self) -> "DSPSignal":
        """Autocorrelation of the signal

        Returns:
          The resulting :class:`blocksim.dsp.DSPSignal`

        """
        ac = self.correlate(self)
        return ac

    def applyFunction(self, fct: Callable) -> "DSPSignal":
        """Applies a function to all the samples

        Args:
          fct
            A callable to apply to all samples

        Returns:
          The resulting :class:`blocksim.dsp.DSPSignal`

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
          The resulting :class:`blocksim.dsp.DSPSignal`

        """
        return self.applyFunction(np.conj)

    def reverse(self) -> "DSPSignal":
        """Returns the reversed signal

        Returns:
          The resulting :class:`blocksim.dsp.DSPSignal`

        """
        return self.applyFunction(lambda x: x[::-1])

    def __matmul__(self, y: "DSPSignal") -> "DSPSignal":
        return self.convolve(y)

    def convolve(self, y: "DSPSignal") -> "DSPSignal":
        """Returns the convolution with another :class:`blocksim.dsp.DSPSignal`

        Args:
          y
            The :class:`blocksim.dsp.DSPSignal` to convolve with

        Returns:
          The resulting :class:`blocksim.dsp.DSPSignal`

        """
        z = y.reverse().conj()
        return self.correlate(z)

    def forceSamplingStart(self, samplingStart: float) -> "DSPLine":
        res = DSPSignal(
            name=self.name,
            samplingStart=samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=self.y_serie,
            default_transform=self.default_transform,
        )
        return res

    def integrate(self, period: float, offset: float = 0) -> "DSPSignal":
        """

        Args:
          period (s)
            Size of the window in the time domain
          offset (s)
            Time of the beginning of the first window.
            Zero means that the first window starts when the signal starts

        """
        otp = self.getOutputByName("setpoint")
        typ = otp.getDataType()

        if (
            typ == np.complex128
            or typ == np.complex160
            or typ == np.complex192
            or typ == np.complex256
            or typ == np.complex512
            or typ == np.complex64
        ):
            comp_out = True
        else:
            comp_out = False

        s_start = self.samplingStart
        s_stop = self.samplingStop
        w_start = offset + s_start
        dt = self.samplingPeriod

        w_len = int(period / dt + 1)

        n_win = int((s_stop - s_start) / period)

        res = DSPSignal(
            name=self.name,
            samplingStart=0,
            samplingPeriod=dt,
            y_serie=np.zeros(w_len, dtype=np.complex),
            default_transform=self.default_transform,
        )
        for k in range(n_win):
            chunk = self.resample(
                samplingStart=w_start + k * period,
                samplingPeriod=dt,
                samplingStop=w_start + (k + 1) * period,
                complex_output=self.hasOutputComplex,
            )
            res = res + chunk.forceSamplingStart(0) / n_win

        return res
