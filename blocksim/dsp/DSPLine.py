from typing import Callable

# from abc import ABCMeta

import numpy as np
from scipy.interpolate import interp1d


__all__ = ["DSPLine"]


class DSPLine(object):
    """Generic DSP line

    Args:
      name
        Name of the line
      samplingStart
        First x coord of the sample of the line
      samplingPeriod (Hz)
        x coord spacing of the line
      y_serie
        Complex samples of the line
      default_transform
        Function to apply to the samples before plotting.
        Shall be vectorized

    """

    name_of_x_var = "Samples"
    unit_of_x_var = "ech"

    def __init__(
        self,
        name: str,
        samplingStart: float = None,
        samplingPeriod: float = None,
        y_serie: np.array = None,
        default_transform=lambda x: x,
    ):
        self.name = name
        self.samplingStart = samplingStart
        self.samplingPeriod = samplingPeriod
        self.y_serie = y_serie
        self.default_transform = default_transform

    def generateXSerie(self, index: int = None) -> np.array:
        """Generates the x samples of the line

        Args:
          index
            If given, returns only the x coord at the position given by index

        Returns:
          The x coordinate(s)

        """
        if index is None:
            n = len(self)
            index = np.arange(n)
        x = index * self.samplingPeriod + self.samplingStart
        return x

    def findPeaksWithTransform(
        self, transform: Callable = None, nb_peaks: int = 3
    ) -> np.array:
        """Finds the peaks in a :class:`SystemControl.dsp.DSPLine`.
        The search is performed on the tranformed samples (with the argument *transform*, or the attribute *default_transform*)

        Args:
          transform
            A callable applied on samples before looking for the peaks
          nb_peaks
            Max number of peaks to seach. Only the highest are kept

        Returns:
          The x coordinate(s) of the peaks

        """
        if transform is None:
            transform = self.default_transform

        dat = transform(self.y_serie)
        n = len(dat)
        lpeak = []
        for i in range(1, n - 1):
            if dat[i - 1] < dat[i] and dat[i] > dat[i + 1]:
                lpeak.append((i, dat[i]))

        lpeak.sort(key=lambda x: x[1], reverse=True)

        if len(lpeak) > nb_peaks:
            lpeak = lpeak[:nb_peaks]

        return np.array([self.generateXSerie(x) for x, y in lpeak])

    def getSample(self, x: float) -> np.complex128:
        """Gets the sample at x-coord x. A cubic interpolation is used.

        Args:
          x
            The x-coord of the sample to retrieve

        Returns:
          The complex sample

        """
        res = self.resample(samplingStart=x, samplingPeriod=1, samplingStop=x)
        return res[0]

    @property
    def samplingStop(self) -> float:
        """Gets the last x-coord of the line

        Returns:
          Last x-coord of the line

        """
        return self.samplingStart + (len(self) - 1) * self.samplingPeriod

    def resample(
        self, samplingStart: float, samplingPeriod: float, samplingStop: float = None
    ) -> "DSPLine":
        """Resamples the line using a cubic interpolation.
        If the new bounds are larger than the original ones, the line is filled with 0
        The resulting signal keeps the timestamp of samples

        Args:
          samplingStart
            First x-coord of the sample of the line after resampling
          samplingPeriod
            x-coord spacing of the line after resampling
          samplingStop
            Last x-coord of the sample of the line after resampling

        """
        if samplingStop is None:
            samplingStop = self.samplingStop

        ns = int(np.round((samplingStop - samplingStart) / samplingPeriod, 0)) + 1
        new_x = np.arange(ns) * samplingPeriod + samplingStart

        if len(self) == 1:
            kind = "nearest"
        elif len(self) == 2:
            kind = "linear"
        else:
            kind = "cubic"

        itp_x = interp1d(
            self.generateXSerie(),
            np.real(self.y_serie),
            kind=kind,
            copy=False,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )
        itp_y = interp1d(
            self.generateXSerie(),
            np.imag(self.y_serie),
            kind=kind,
            copy=False,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )

        y_serie = 1j * itp_y(new_x)
        y_serie += itp_x(new_x)

        return self.__class__(
            name=self.name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )

    @classmethod
    def to_db(cls, x: np.array, low: float = -80) -> np.array:
        """Converts the samples into their power, in dB.
        If a sample's power is below *low*, the dB value in clamped to *low*.

        Args:
          x
            The array of samples
          low (dB)
            The min value to clamp to

        Returns:
          The power of the serie *x* in dB

        """
        low_lin = 10 ** (low / 10)
        amp = np.real(np.conj(x) * x)
        if hasattr(amp, "__iter__"):
            i_th = np.where(amp < low_lin)[0]
            amp[i_th] = low_lin
        else:
            amp = max(low_lin, amp)

        return 10 * np.log10(amp)

    @classmethod
    def getUnitAbbrev(cls, mult: float) -> str:
        """Given a scale factor, gives the prefix for the unit to display

        Args:
          mult
            Scale factor

        Returns:
          Prefix

        """
        d = {
            1: "",
            1000: "k",
            1e6: "M",
            1e9: "G",
            1e12: "T",
            1e-3: "m",
            1e-6: "µ",
            1e-9: "n",
            1e-12: "p",
        }
        return d[mult]

    def __len__(self):
        return len(self.y_serie)

    def __iter__(self):
        for y in self.y_serie:
            yield y

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError(key)

        y = self.y_serie[key]

        return y

    def __radd__(self, y) -> "DSPLine":
        return self + y

    def __add__(self, y: "DSPLine") -> "DSPLine":
        if issubclass(y.__class__, DSPLine):
            t_start = min(self.samplingStart, y.samplingStart)
            dt = min(self.samplingPeriod, y.samplingPeriod)
            t_stop = max(self.samplingStop, y.samplingStop)

            rx = self.resample(
                samplingStart=t_start, samplingPeriod=dt, samplingStop=t_stop
            )
            ry = y.resample(
                samplingStart=t_start, samplingPeriod=dt, samplingStop=t_stop
            )

            y_serie = rx.y_serie + ry.y_serie

        else:
            t_start = self.samplingStart
            dt = self.samplingPeriod
            y_serie = self.y_serie + y

        return self.__class__(
            name=self.name,
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )

    def __neg__(self) -> "DSPLine":
        return self.__class__(
            name=self.name,
            samplingStart=self.samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=-self.y_serie,
            default_transform=self.default_transform,
        )

    def __rsub__(self, y) -> "DSPLine":
        z = -self
        return y + z

    def __sub__(self, y: "DSPLine") -> "DSPLine":
        z = -y
        return self + z

    def __rmul__(self, y):
        return self * y

    def __truediv__(self, y):
        if not issubclass(y.__class__, DSPLine):
            t_start = self.samplingStart
            dt = self.samplingPeriod
            y_serie = self.y_serie / y

        return self.__class__(
            name=self.name,
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )

    def __mul__(self, y: "DSPLine") -> "DSPLine":
        if issubclass(y.__class__, DSPLine):
            t_start = min(self.samplingStart, y.samplingStart)
            dt = min(self.samplingPeriod, y.samplingPeriod)
            t_stop = max(self.samplingStop, y.samplingStop)

            rx = self.resample(
                samplingStart=t_start, samplingPeriod=dt, samplingStop=t_stop
            )
            ry = y.resample(
                samplingStart=t_start, samplingPeriod=dt, samplingStop=t_stop
            )

            y_serie = rx.y_serie * ry.y_serie

        else:
            t_start = self.samplingStart
            dt = self.samplingPeriod
            y_serie = self.y_serie * y

        return self.__class__(
            name=self.name,
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )
