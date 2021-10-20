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

    # __slots__=["__name","__samplingStart","__samplingPeriod","__y_serie","__default_transform"]

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
        self.__name = name
        self.__samplingStart = samplingStart
        self.__samplingPeriod = samplingPeriod
        self.__y_serie = y_serie
        self.__default_transform = default_transform

        self.__bootstrap()

    @property
    def name(self) -> str:
        return self.__name[:]

    @property
    def samplingStart(self) -> str:
        return self.__samplingStart

    @property
    def samplingPeriod(self) -> str:
        return self.__samplingPeriod

    @property
    def y_serie(self) -> np.array:
        return self.__y_serie.copy()

    @property
    def default_transform(self) -> np.array:
        return self.__default_transform

    def __bootstrap(self):
        size = len(self)

        index = np.arange(size)
        self.__x_serie = index * self.samplingPeriod + self.samplingStart

        if size == 1:
            kind = "nearest"
        elif size == 2:
            kind = "linear"
        else:
            kind = "cubic"

        self.__itp_x = interp1d(
            self.__x_serie,
            np.real(self.y_serie),
            kind=kind,
            copy=False,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )

        self.__itp_y = interp1d(
            self.__x_serie,
            np.imag(self.y_serie),
            kind=kind,
            copy=False,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )

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
            index = range(n)
        x = self.__x_serie[index]
        return x

    def findPeaksWithTransform(
        self, transform: Callable = None, nb_peaks: int = 3
    ) -> np.array:
        """Finds the peaks in a :class:`blocksim.dsp.DSPLine`.
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

    def __interpolate(self, new_x: np.array, complex_output: bool = True) -> np.array:
        if complex_output:
            y_serie = 1j * self.__itp_y(new_x)
            y_serie += self.__itp_x(new_x)
        else:
            y_serie = self.__itp_x(new_x)

        return y_serie

    def getSample(self, x: float, complex_output: bool = True) -> np.complex128:
        """Gets the sample at x-coord x. A cubic interpolation is used.

        Args:
          x
            The x-coord of the sample to retrieve
          complex_output
            True if we interpolate both real part and imag part

        Returns:
          The real (resp. complex) sample if complex_output is False (resp. True)

        """
        res = self.__interpolate(x, complex_output=complex_output)

        return res

    @property
    def samplingStop(self) -> float:
        """Gets the last x-coord of the line

        Returns:
          Last x-coord of the line

        """
        return self.samplingStart + (len(self) - 1) * self.samplingPeriod

    def truncate(self, samplingStart: float, samplingStop: float) -> "DSPLine":
        """Truncates a line between x=samplingStart and x=samplingStop.

        Args:
          samplingStart
            Beginning abscissa of the new line
          samplingStop
            End abscissa of the new line

        Returns:
          A truncated new line with the same spacing

        """
        istart = np.where(self.__x_serie >= samplingStart)[0]
        iend = np.where(self.__x_serie < samplingStop)[0]
        iok = np.intersect1d(istart, iend)
        if len(iok) == 0:
            y_serie = np.array([], dtype=np.complex128)
        else:
            samplingStart = self.generateXSerie(iok[0])
            y_serie = self.y_serie[iok]

        return self.__class__(
            name=self.name,
            samplingStart=samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )

    def resample(
        self,
        samplingStart: float,
        samplingPeriod: float = None,
        samplingStop: float = None,
        complex_output: bool = True,
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
          complex_output
            True if we interpolate both real part and imag part

        """
        if samplingStop is None:
            samplingStop = self.samplingStop

        if samplingPeriod is None:
            samplingPeriod = self.samplingPeriod

        ns = int(np.round((samplingStop - samplingStart) / samplingPeriod, 0)) + 1
        new_x = np.arange(ns) * samplingPeriod + samplingStart

        y_serie = self.__interpolate(new_x, complex_output=complex_output)

        return self.__class__(
            name=self.name,
            samplingStart=samplingStart,
            samplingPeriod=samplingPeriod,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )

    @classmethod
    def to_db_lim(cls, low: float) -> Callable:
        """Returns a function that turns a complex signal into the power serie of the signal, in dB.

        Args:
          low (dB)
            The min value to clamp to

        Returns:
          The function to map on a complex time serie

        Examples:
          >>> f = DSPLine.to_db_lim(low=-80)
          >>> f(1e-3)
          -60.0
          >>> f(1e-4)
          -80.0
          >>> f(1e-5)
          -80.0

        """

        def _to_db(x):
            low_lin = 10 ** (low / 10)
            amp = np.real(np.conj(x) * x)
            if hasattr(amp, "__iter__"):
                i_th = np.where(amp < low_lin)[0]
                amp[i_th] = low_lin
            else:
                amp = max(low_lin, amp)

            return 10 * np.log10(amp)

        return _to_db

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
