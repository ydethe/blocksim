import numpy as np
from numpy.fft import ifft, fftshift
from scipy.signal import get_window

from .DSPLine import DSPLine


__all__ = ["DSPSpectrum"]


class DSPSpectrum(DSPLine):
    """Spectrum of a signal

    Args:
      name
        Name of the spectrum
      samplingStart (Hz)
        First frequency of the sample of the spectrum
      samplingPeriod (Hz)
        Frequency spacing of the spectrum
      y_serie
        Complex samples of the spectrum
      default_transform
        Function to apply to the samples before plotting.
        Shall be vectorized

    """

    def __init__(
        self,
        name: str,
        samplingStart: float = None,
        samplingPeriod: float = None,
        y_serie: np.array = None,
        default_transform=np.abs,
    ):
        DSPLine.__init__(
            self,
            name=name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=y_serie,
            default_transform=default_transform,
        )

        self.name_of_x_var = "Frequency"
        self.unit_of_x_var = "Hz"

    @property
    def energy(self) -> float:
        return np.real(self.y_serie @ self.y_serie.conj() * len(self))

    def ifft(self, win: str = "ones") -> "DSPSignal":
        """Applies the inverse discrete Fourier transform

        Args:
          win
            The window to be applied. Should be compatible with `get_window`_.

            .. _get_window: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html

        Returns:
          The resulting :class:`blocksim.dsp.DSPSignal`

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
