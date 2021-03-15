import numpy as np
from numpy import log10, exp, pi, sqrt, cos, sin
from scipy.signal import firwin2, firwin, lfilter_zi, lfilter
from numpy.polynomial import Polynomial

from .DSPLine import DSPLine
from .DSPSignal import DSPSignal
from .utils import phase_unfold


__all__ = ["DSPFilter"]


class DSPFilter(object):
    """A filter

    Args:
      name
        Name of the spectrum
      f_low (Hz)
        Start frequency of the band pass
      f_high (Hz)
        End frequency of the band pass
      numtaps
        Number of coefficients
      win
        The window to be applied. Should be compatible with `get_window`_.

        .. _get_window: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html

    """

    def __init__(
        self,
        name: str,
        f_low: float,
        f_high: float,
        numtaps: int,
        win: str = "hamming",
    ):
        self.f_low = f_low
        self.f_high = f_high
        self.numtaps = numtaps
        self.win = win

    def generateCoefficients(self, fs: float) -> np.array:
        """Generates the filter's coefficients

        Args:
          fs (Hz)
            The sampling frequency of the signals that will be filtered through it

        Returns:
          The coefficients

        """
        # https://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need/31077
        d = 10e-2
        nt = int(-2 / 3 * log10(10 * d ** 2) * fs / (self.f_high - self.f_low))
        if nt > self.numtaps:
            raise ValueError(self.numtaps, nt)

        b = firwin(
            self.numtaps,
            [self.f_low, self.f_high],
            pass_zero=False,
            scale=True,
            window=self.win,
            fs=fs,
        )

        return b

    def plotBode(self, fs: float, axe_amp, axe_pha):
        """Plots the bode diagram of the filter

        Args:
          fs (Hz)
            The sampling frequency of the signals that will be filtered through it
          axe_amp
            Matplotlib axe to draw the ampltiude on
          axe_pha
            Matplotlib axe to draw the unfolded phase on

        """
        n = 200
        b = self.generateCoefficients(fs)

        freq = np.linspace(0, fs / 2, n)

        p = Polynomial(b)
        z = exp(-1j * 2 * pi * freq / fs)
        y = p(z)

        axe_amp.plot(freq, DSPLine.to_db(y))
        axe_amp.grid(True)
        axe_amp.set_ylabel("Ampliude (dB)")

        pha = phase_unfold(y)

        axe_pha.plot(freq, 180 / pi * pha)
        axe_pha.grid(True)
        axe_pha.set_xlabel("Frequency (Hz)")
        axe_pha.set_ylabel("Phase (deg)")

    def apply(self, s: DSPSignal) -> DSPSignal:
        """Filters a :class:`SystemControl.dsp.DSPSignal`

        Args:
          s
            The :class:`SystemControl.dsp.DSPSignal` to filter

        Returns:
          The filtered signal

        """
        fs = 1 / s.samplingPeriod
        b = self.generateCoefficients(fs)

        zi = lfilter_zi(b, [1])
        z, _ = lfilter(b, [1], s.y_serie, zi=zi * s.y_serie[0])

        return DSPSignal(
            name="Filtered %s" % s.name,
            samplingStart=s.samplingStart - self.numtaps / fs / 2,
            samplingPeriod=s.samplingPeriod,
            y_serie=z,
            default_transform=np.real,
        )
