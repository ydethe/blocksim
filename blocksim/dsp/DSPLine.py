from typing import Callable, List
from itertools import product
from math import factorial

from lazy_property import LazyProperty
import numpy as np
from scipy import linalg as lin
from numpy.lib.arraysetops import isin
from scipy.interpolate import interp1d

from . import derivative_coeff
from .. import logger
from .Peak import Peak

__all__ = ["DSPLine"]


class DSPLine(object):
    """Generic DSP line

    Args:
        name: Name of the line
        samplingStart: First x coord of the sample of the line
        samplingPeriod: x coord spacing of the line
        y_serie: Complex samples of the line
        projection: Axe projection. Can be 'rectilinear' or 'polar'
        default_transform: Function to apply to the samples before plotting.
            Shall be vectorized

    """

    # __slots__=["__name","__samplingStart","__samplingPeriod","__y_serie","__default_transform"]

    def __init__(
        self,
        name: str,
        samplingStart: float = None,
        samplingPeriod: float = None,
        y_serie: np.array = None,
        projection: str = "rectilinear",
        default_transform=lambda x: x,
    ):
        self.__name = name
        self.__samplingStart = samplingStart
        self.__samplingPeriod = samplingPeriod
        self.__y_serie = y_serie
        self.__default_transform = default_transform
        self.name_of_x_var = "Samples"
        self.unit_of_x_var = "ech"
        self.projection = projection

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

    @LazyProperty
    def _itp_x(self):
        size = len(self)

        if size == 1:
            kind = "nearest"
        elif size == 2:
            kind = "linear"
        else:
            kind = "cubic"

        itp = interp1d(
            self._x_serie,
            np.real(self.y_serie),
            kind=kind,
            copy=False,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )

        return itp

    @LazyProperty
    def _itp_y(self):
        size = len(self)

        if size == 1:
            kind = "nearest"
        elif size == 2:
            kind = "linear"
        else:
            kind = "cubic"

        itp = interp1d(
            self._x_serie,
            np.imag(self.y_serie),
            kind=kind,
            copy=False,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )

        return itp

    @LazyProperty
    def _x_serie(self):
        size = len(self)

        index = np.arange(size)
        x_serie = index * self.samplingPeriod + self.samplingStart

        return x_serie

    def generateXSerie(self, index: int = None) -> np.array:
        """Generates the x samples of the line

        Args:
            index: If given, returns only the x coord at the position given by index

        Returns:
            The x coordinate(s)

        """
        n = len(self)
        if index is None:
            index = np.arange(n)
        elif hasattr(index, "__iter__"):
            index = np.array(index)
        elif index < 0:
            index += n
        x = index * self.samplingPeriod + self.samplingStart
        return x

    def findPeaksWithTransform(
        self, transform: Callable = None, nb_peaks: int = 3
    ) -> List[Peak]:
        """Finds the peaks in a :class:`blocksim.dsp.DSPLine`.
        The search is performed on the tranformed samples (with the argument *transform*, or the attribute *default_transform*)

        Args:
            transform: A callable applied on samples before looking for the peaks
            nb_peaks: Max number of peaks to seach. Only the highest are kept

        Returns:
            The list of detected peaks, sorted by descreasing value of the peak

        """
        if transform is None:
            transform = self.default_transform

        dat = transform(self.y_serie)
        n = len(dat)
        lpeak = []
        ep = 1
        for p0 in range(ep, n - ep):
            if dat[p0 - ep] < dat[p0] and dat[p0] > dat[p0 + ep]:
                b = (dat[p0 + ep] - dat[p0 - ep]) / (2 * ep)
                c = -(-dat[p0 + ep] - dat[p0 - ep] + 2 * dat[p0]) / (2 * ep**2)
                dp = -b / (2 * c)
                dval = -(b**2) / (4 * c)
                x0 = self.generateXSerie(p0 + dp)
                p = Peak(
                    coord_label=(self.name_of_x_var,),
                    coord_unit=(self.unit_of_x_var,),
                    coord=(x0,),
                    value=dat[p0] + dval,
                )
                lpeak.append(p)

        lpeak.sort(key=lambda x: x.value, reverse=True)

        if len(lpeak) > nb_peaks:
            lpeak = lpeak[:nb_peaks]

        return lpeak

    def __interpolate(self, new_x: np.array, complex_output: bool = True) -> np.array:
        if complex_output:
            y_serie = 1j * self._itp_y(new_x)
            y_serie += self._itp_x(new_x)
        else:
            y_serie = self._itp_x(new_x)

        return y_serie

    def __call__(self, x: float):
        return self.getSample(x, complex_output=True)

    def getSample(self, x: float, complex_output: bool = True) -> np.complex128:
        """Gets the sample at x-coord x. A cubic interpolation is used.

        Args:
            x: The x-coord of the sample to retrieve
            complex_output: True if we interpolate both real part and imag part

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

    def truncate(
        self,
        samplingStart: float = None,
        samplingStop: float = None,
        zero_padding: bool = True,
    ) -> "DSPLine":
        """Truncates a line between x=samplingStart and x=samplingStop.

        Args:
            samplingStart: Beginning abscissa of the new line
            samplingStop: End abscissa of the new line (included)
            zero_padding: Add zero padding if samplingStart or samplingStop are beyond the bounds of the signal

        Returns:
            A truncated new line with the same spacing

        """
        if samplingStart is None:
            samplingStart = self.samplingStart

        if samplingStop is None:
            ns = len(self)
            samplingStop = self.generateXSerie(ns - 1)

        dt = self.samplingPeriod
        istart = int(np.ceil((samplingStart - self.samplingStart) / dt))
        iend = int(np.floor((samplingStop - self.samplingStart) / dt)) + 1

        nzl = max(-istart, 0)
        nzr = max(iend - len(self), 0)
        if zero_padding and (nzl > 0 or nzr > 0):
            yp = np.pad(array=self.y_serie, pad_width=(nzl, nzr), mode="constant")
            samplingStart = self.samplingStart - nzl * dt
        elif not zero_padding and (nzl > 0 or nzr > 0):
            raise IndexError("Index out of range")
        elif not zero_padding and nzl == 0 and nzr == 0:
            iok = slice(istart, iend)
            yp = self.y_serie[iok]
            samplingStart = self.generateXSerie(istart)
        elif zero_padding and nzl == 0 and nzr == 0:
            iok = slice(istart, iend)
            yp = self.y_serie[iok]
            samplingStart = self.generateXSerie(istart)

        return self.__class__(
            name=self.name,
            samplingStart=samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=yp,
            default_transform=self.default_transform,
        )

    def isInSyncWith(self, y) -> bool:
        """Tests whether the line is synced with y.
        y can be either:

        * a scalar (float or int). In this case, y is a samplingPeriod and the test is successful if 
            $$ |self.samplingPeriod - y| < 10^-6 * min(self.samplingPeriod, y) $$

        Args:
            y: The description of the serie to check
        
        Returns:
            The result of the test

        """
        if isinstance(y, float) or isinstance(y, int):
            dty = y
            t0y = self.generateXSerie(0)
        elif isinstance(y, tuple):
            t0y, dty = y
        else:
            dty = y.samplingPeriod
            t0y = y.generateXSerie(0)

        dt = min(self.samplingPeriod, dty)
        k_time, _ = np.modf((self.generateXSerie(0) - t0y) / dt)

        err_dt = np.abs(self.samplingPeriod - dty) / dt

        return err_dt < 1e-6 and k_time < 1e-3

    def resample(
        self,
        samplingStart: float,
        samplingPeriod: float = None,
        samplingStop: float = None,
        nech: int = None,
        zero_padding: bool = True,
        complex_output: bool = None,
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
            Last x-coord (included) of the sample of the line after resampling
          complex_output
            True if we interpolate both real part and imag part

        """
        if complex_output is None:
            typ = self.y_serie.dtype
            complex_output = typ == np.complex128 or typ == np.complex64

        if samplingPeriod is None:
            samplingPeriod = self.samplingPeriod

        if samplingStop is None and nech is None:
            samplingStop = self.samplingStop
            nech = 1 + int(np.ceil((samplingStop - samplingStart) / samplingPeriod))
        elif not samplingStop is None and nech is None:
            nech = 1 + int(np.ceil((samplingStop - samplingStart) / samplingPeriod))
        elif samplingStop is None and not nech is None:
            samplingStop = (nech - 1) * samplingPeriod + samplingStart
        else:
            raise AssertionError("nech and samplingStop cannot be set simultaneously")

        if self.isInSyncWith((samplingStart, samplingPeriod)):
            logger.debug("Truncate '%s'" % self.name)
            res = self.truncate(
                samplingStart=samplingStart,
                samplingStop=samplingStop + samplingPeriod / 2,
                zero_padding=zero_padding,
            )
            if len(res) != nech:
                raise AssertionError("Wrong truncation: %i, %i" % (len(res), nech))
        else:
            logger.debug("Resample '%s'" % self.name)
            new_x = np.arange(nech) * samplingPeriod + samplingStart

            y_serie = self.__interpolate(new_x, complex_output=complex_output)

            nz = nech - len(y_serie)
            if zero_padding and nz > 0:
                y_serie = np.pad(array=y_serie, pad_width=(0, nz), mode="constant")

            res = self.__class__(
                name=self.name,
                samplingStart=samplingStart,
                samplingPeriod=samplingPeriod,
                y_serie=y_serie,
                default_transform=self.default_transform,
            )
            if len(res) != nech:
                raise AssertionError("Wrong resampling: %i, %i" % (len(res), nech))

        return res

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
            pwr = np.real(np.conj(x) * x)
            pwr = np.clip(pwr, low_lin, None)
            return 10 * np.log10(pwr)

        return _to_db

    @classmethod
    def to_db(cls, x: "array") -> np.array:
        """Converts the samples into their power, in dB.
        If a sample's power is below *low*, the dB value in clamped to *low*.

        Args:
          x
            The array of samples

        Returns:
          The power of the serie *x* in dB

        """
        pwr = np.real(np.conj(x) * x)
        return 10 * np.log10(pwr)

    def __len__(self):
        return len(self.y_serie)

    def __iter__(self):
        for y in self.y_serie:
            yield y

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            pass
        elif isinstance(idx, int):
            lid = idx
        else:
            raise TypeError("Invalid argument type.")

        return self.y_serie[lid]

    def derivate(self, rank: int = 1, order: int = None) -> "DSPLine":
        """Performs numerical derivation if the line

        Args:
            rank
                Rank of the derivative
            order
                Order of the Taylor serie used to estimate the derivative. Shall be >= rank
                The default value is *rank*

        """
        coeffs = derivative_coeff(rank, order)
        n = len(coeffs)
        k = (n - 1) // 2

        ns = len(self) - 2 * k
        dy = np.zeros(ns, dtype=self.y_serie.dtype)
        for p in range(-k, k + 1):
            dy += coeffs[p + k] * self.y_serie[p + k : ns + p + k]

        return self.__class__(
            name=self.name,
            samplingStart=self.samplingStart + k * self.samplingPeriod,
            samplingPeriod=self.samplingPeriod,
            y_serie=dy / self.samplingPeriod**rank,
            default_transform=self.default_transform,
        )

    def _prepareOperation(self, y: "DSPLine"):
        t_start = min(self.samplingStart, y.samplingStart)
        dt = min(self.samplingPeriod, y.samplingPeriod)
        t_stop = max(self.samplingStop, y.samplingStop)
        nech = int((t_stop - t_start) / dt + 1)

        rx = self.resample(samplingStart=t_start, samplingPeriod=dt, nech=nech)
        ry = y.resample(samplingStart=t_start, samplingPeriod=dt, nech=nech)

        return t_start, dt, rx, ry

    def __radd__(self, y) -> "DSPLine":
        return self + y

    def __add__(self, y: "DSPLine") -> "DSPLine":
        if issubclass(y.__class__, DSPLine):
            t_start, dt, rx, ry = self._prepareOperation(y)

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
        else:
            t_start, dt, rx, ry = self._prepareOperation(y)
            y_serie = rx.y_serie / ry.y_serie

        return self.__class__(
            name=self.name,
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )

    def __mul__(self, y: "DSPLine") -> "DSPLine":
        if issubclass(y.__class__, DSPLine):
            t_start, dt, rx, ry = self._prepareOperation(y)

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
