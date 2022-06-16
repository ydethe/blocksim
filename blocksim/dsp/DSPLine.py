from enum import Enum
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Any

from lazy_property import LazyProperty
from nptyping import NDArray
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy import linalg as lin

from . import derivative_coeff
from .. import logger
from ..utils import find1dpeak, Peak
from ..graphics.GraphicSpec import AxeProjection, DSPLineType

__all__ = ["DSPRectilinearLine", "DSPPolarLine", "DSPNorthPolarLine"]


class ADSPLine(metaclass=ABCMeta):
    """Generic DSP line

    Args:
        name: Name of the line
        samplingStart: First x coord of the sample of the line
        samplingPeriod: x coord spacing of the line
        y_serie: Complex samples of the line
        default_transform: Function to apply to the samples before plotting.
            Shall be vectorized

    """

    # Has to be commented because of multiple inheritance
    # __slots__=["__name","__samplingStart","__samplingPeriod","__y_serie","__default_transform"]

    def __init__(
        self,
        name: str,
        samplingStart: float = None,
        samplingPeriod: float = None,
        y_serie: NDArray[Any, Any] = None,
        default_transform=lambda x: x,
    ):
        self.__name = name
        self.__samplingStart = samplingStart
        self.__samplingPeriod = samplingPeriod
        self.__y_serie = y_serie
        self.__default_transform = default_transform
        self.name_of_x_var = "Samples"
        self.unit_of_x_var = "ech"

    @property
    @abstractmethod
    def dspline_type(self) -> DSPLineType:
        pass

    @property
    def name(self) -> str:
        return self.__name[:]

    @property
    def samplingStart(self) -> float:
        return self.__samplingStart

    @property
    def samplingPeriod(self) -> float:
        return self.__samplingPeriod

    @property
    def y_serie(self) -> NDArray[Any, Any]:
        return self.__y_serie.copy()

    @property
    def default_transform(self):
        return self.__default_transform

    @LazyProperty
    def _itp_x(self):
        # size = len(self)
        size = 2

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
        # size = len(self)
        size = 2

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

    def generateXSerie(self, index: int = None) -> NDArray[Any, Any]:
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

    def getAdaptedAxeProjection(self) -> AxeProjection:
        if self.dspline_type == DSPLineType.RECTILINEAR:
            return AxeProjection.RECTILINEAR
        elif self.dspline_type == DSPLineType.POLAR:
            return AxeProjection.POLAR
        elif self.dspline_type == DSPLineType.NORTH_POLAR:
            return AxeProjection.NORTH_POLAR

    def quickPlot(self, **kwargs) -> "AxesSubplot":
        """Quickly plots the line on a matplotlib axe

        Args:
            kwargs: Plotting options

        Returns:
            The Axes used by matplotlib

        """
        from ..graphics import quickPlot

        axe = quickPlot(self, **kwargs)
        return axe

    def polyfit(self, deg: int) -> Polynomial:
        """Fits a polynomial to the DSPRectilinearLine. All the samples are considered in the computation.
        See https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html
        for the Polynomial object documentation.

        Args:
            deg: Degree of the fitted polynomial

        Returns:
            A numpy Polynomial

        Examples:
            >>> a = 2 - 3j
            >>> b = -4 + 1j
            >>> x = np.arange(10)
            >>> y = a * x + b
            >>> line = DSPRectilinearLine(name="line", samplingStart=0, samplingPeriod=1, y_serie=y)
            >>> p = line.polyfit(deg=1)
            >>> p
            Polynomial([-4.+1.j,  2.-3.j], domain=[0., 9.], window=[0., 9.])

        """
        x = self.generateXSerie()
        x0 = x[0]
        x1 = x[-1]

        A = np.vander(x, N=deg + 1, increasing=True)
        iA = lin.pinv(A)
        c = iA @ self.y_serie

        return Polynomial(coef=c, domain=[x0, x1], window=[x0, x1])

    def applyDelay(self, delay: float) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        """Applies a delay (in X axis unit) to the DSPRectilinearLine

        Args:
            delay: Delay to be applied

        Returns:
            The delayed DSPRectilinearLine

        """
        t_start = self.samplingStart + delay
        ech = self.y_serie

        dsig = self.__class__(
            name=self.name,
            samplingStart=t_start,
            samplingPeriod=self.samplingPeriod,
            y_serie=ech,
            default_transform=self.default_transform,
        )
        dsig.name_of_x_var = self.name_of_x_var
        dsig.unit_of_x_var = self.unit_of_x_var
        return dsig

    def repeat(self, repeats: int) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        """Repeats the samples *repeats* time

        Args:
            repeats: Number of repetitions

        Returns:
            The repeated DSPRectilinearLine

        """
        dsig = self.__class__(
            name=self.name,
            samplingStart=self.samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=np.tile(self.y_serie, reps=repeats),
            default_transform=self.default_transform,
        )
        dsig.name_of_x_var = self.name_of_x_var
        dsig.unit_of_x_var = self.unit_of_x_var
        return dsig

    def findPeaksWithTransform(
        self, transform: Callable = None, nb_peaks: int = 3
    ) -> List[Peak]:
        """Finds the peaks in a DSPRectilinearLine.
        The search is performed on the transformed samples (with the argument *transform*, or the attribute *default_transform*)

        Args:
            transform: A callable applied on samples before looking for the peaks
            nb_peaks: Max number of peaks to seach. Only the highest are kept

        Returns:
            The list of detected peaks, sorted by descreasing value of the peak

        """
        if transform is None:
            transform = self.default_transform

        x0 = self.generateXSerie()
        dat = transform(self.y_serie)

        lpeak = find1dpeak(
            nb_peaks=nb_peaks,
            xd=x0,
            yd=dat,
            name_of_x_var=self.name_of_x_var,
            unit_of_x_var=self.unit_of_x_var,
        )

        return lpeak

    def __interpolate(
        self, new_x: NDArray[Any, Any], complex_output: bool = True
    ) -> NDArray[Any, Any]:
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
    ) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
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

        res = self.__class__(
            name=self.name,
            samplingStart=samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=yp,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var
        return res

    def isInSyncWith(self, y) -> bool:
        """Tests whether the line is synced with y.
        In the following, we note dt the sampling period (self.samplingPeriod) and t0 the initial timestamp (self.samplingStart)
        y can be either:

        * a scalar (float or int). In this case, y is a samplingPeriod and the test is successful if:
            $$ |dt - y| < 10^-6 * min(dt, y) $$
        * a tuple of 2 scalars. In this case, y is noted (t0y, dty) which respectively stand for an initial time and a samplingPeriod.
            The test is successful if:
            $$ |dt - dty| < 10^-6 * min(dt, dty) $$
            $$ modf((t0 - t0y) / dt) < 10^-3 $$
        * a DSPRectilinearLine. In this case, t0y is y.samplingStart and dty is y.samplingPeriod
            The test is successful if:
            $$ |dt - dty| < 10^-6 * min(dt, dty) $$
            $$ modf((t0 - t0y) / dt) < 10^-3 $$

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
    ) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        """Resamples the line using a cubic interpolation.
        If the new bounds are larger than the original ones, the line is filled with 0
        The resulting signal keeps the timestamp of samples

        Args:
            samplingStart: First x-coord of the sample of the line after resampling
            samplingPeriod: x-coord spacing of the line after resampling
            samplingStop: Last x-coord (included) of the sample of the line after resampling
            complex_output: True if we interpolate both real part and imag part

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

        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var

        return res

    @classmethod
    def to_db_lim(cls, low: float) -> Callable:
        """Returns a function that turns a complex signal into the power serie of the signal, in dB.

        Args:
            low: The min value to clamp to (dB)

        Returns:
            The function to map on a complex time serie

        Examples:
            >>> f = DSPRectilinearLine.to_db_lim(low=-80)
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
    def to_db(cls, x: NDArray[Any, Any], lim_db: float = -100) -> NDArray[Any, Any]:
        """Converts the samples into their power, in dB.
        If a sample's power is below *low*, the dB value in clamped to *low*.

        Args:
            x: The array of samples
            lim_db: Limit to clamp the power (dB)
                Pass None to avoid any clipping

        Returns:
            The power of the serie *x* (dB)

        """
        lim_lin = 10 ** (lim_db / 10)
        pwr = np.real(np.conj(x) * x)
        pwr = np.clip(pwr, a_min=lim_lin, a_max=None)
        pwr_db = 10 * np.log10(pwr)
        return pwr_db

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

    def derivate(
        self, rank: int = 1, order: int = None
    ) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        """Performs numerical derivation if the line

        Args:
            rank: Rank of the derivative
            order: Order of the Taylor serie used to estimate the derivative. Shall be >= rank
                The default value is *rank*

        """
        coeffs = derivative_coeff(rank, order)
        n = len(coeffs)
        k = (n - 1) // 2

        ns = len(self) - 2 * k
        dy = np.zeros(ns, dtype=self.y_serie.dtype)
        for p in range(-k, k + 1):
            dy += coeffs[p + k] * self.y_serie[p + k : ns + p + k]

        res = self.__class__(
            name=self.name,
            samplingStart=self.samplingStart + k * self.samplingPeriod,
            samplingPeriod=self.samplingPeriod,
            y_serie=dy / self.samplingPeriod**rank,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var
        return res

    def _prepareOperation(self, y: "blocksim.dsp.DSPLine.DSPRectilinearLine"):
        t_start = min(self.samplingStart, y.samplingStart)
        dt = min(self.samplingPeriod, y.samplingPeriod)
        t_stop = max(self.samplingStop, y.samplingStop)
        nech = int((t_stop - t_start) / dt + 1)

        rx = self.resample(samplingStart=t_start, samplingPeriod=dt, nech=nech)
        ry = y.resample(samplingStart=t_start, samplingPeriod=dt, nech=nech)

        return t_start, dt, rx, ry

    def __radd__(self, y) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        return self + y

    def __add__(
        self, y: "blocksim.dsp.DSPLine.DSPRectilinearLine"
    ) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        if issubclass(y.__class__, DSPRectilinearLine):
            t_start, dt, rx, ry = self._prepareOperation(y)

            y_serie = rx.y_serie + ry.y_serie

        else:
            t_start = self.samplingStart
            dt = self.samplingPeriod
            y_serie = self.y_serie + y

        res = self.__class__(
            name=self.name,
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var
        return res

    def __neg__(self) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        res = self.__class__(
            name=self.name,
            samplingStart=self.samplingStart,
            samplingPeriod=self.samplingPeriod,
            y_serie=-self.y_serie,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var
        return res

    def __rsub__(self, y) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        z = -self
        return y + z

    def __sub__(
        self, y: "blocksim.dsp.DSPLine.DSPRectilinearLine"
    ) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        z = -y
        return self + z

    def __rmul__(self, y):
        return self * y

    def __truediv__(self, y):
        if issubclass(y.__class__, DSPRectilinearLine):
            t_start, dt, rx, ry = self._prepareOperation(y)
            y_serie = np.empty_like(ry.y_serie, dtype=np.complex128)
            y_serie[:] = 0
            iok = np.where(ry.y_serie != 0)[0]
            y_serie[iok] = rx.y_serie[iok] / ry.y_serie[iok]
        elif hasattr(y, "__iter__"):
            t_start = self.samplingStart
            dt = self.samplingPeriod
            y_serie = np.empty_like(y, dtype=np.complex128)
            y_serie[:] = np.nan
            iok = np.where(y != 0)[0]
            y_serie[iok] = self.y_serie[iok] / y[iok]
        else:
            t_start = self.samplingStart
            dt = self.samplingPeriod
            y_serie = self.y_serie / y

        res = self.__class__(
            name=self.name,
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var
        return res

    def __mul__(
        self, y: "blocksim.dsp.DSPLine.DSPRectilinearLine"
    ) -> "blocksim.dsp.DSPLine.DSPRectilinearLine":
        if issubclass(y.__class__, DSPRectilinearLine):
            t_start, dt, rx, ry = self._prepareOperation(y)

            y_serie = rx.y_serie * ry.y_serie

        else:
            t_start = self.samplingStart
            dt = self.samplingPeriod
            y_serie = self.y_serie * y

        res = self.__class__(
            name=self.name,
            samplingStart=t_start,
            samplingPeriod=dt,
            y_serie=y_serie,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var
        return res


class DSPHistogram(ADSPLine):
    @property
    def dspline_type(self) -> DSPLineType:
        return DSPLineType.HISTOGRAM


class DSPRectilinearLine(ADSPLine):
    @property
    def dspline_type(self) -> DSPLineType:
        return DSPLineType.RECTILINEAR

    def histogram(
        self,
        name: str,
        bins="auto",
        weights=None,
        density=None,
        cumulative: bool = False,
        bin_unit: str = "-",
        bin_name: str = "",
        transform=None,
    ) -> DSPHistogram:
        """Computes the histogram of the time serie"""
        # bins : int or sequence of scalars or str, optional
        #    'auto'
        #        Maximum of the 'sturges' and 'fd' estimators. Provides good
        #        all around performance.
        #    'fd' (Freedman Diaconis Estimator)
        #        Robust (resilient to outliers) estimator that takes into
        #        account data variability and data size.
        #    'doane'
        #        An improved version of Sturges' estimator that works better
        #        with non-normal datasets.
        #    'scott'
        #        Less robust estimator that that takes into account data
        #        variability and data size.
        #    'stone'
        #        Estimator based on leave-one-out cross-validation estimate of
        #        the integrated squared error. Can be regarded as a generalization
        #        of Scott's rule.
        #    'rice'
        #        Estimator does not take variability into account, only data
        #        size. Commonly overestimates number of bins required.
        #    'sturges'
        #        R's default method, only accounts for data size. Only
        #        optimal for gaussian data and underestimates number of bins
        #        for large non-gaussian datasets.
        #    'sqrt'
        #        Square root (of data size) estimator, used by Excel and
        #        other programs for its speed and simplicity.
        # weights : array_like, optional
        #     An array of weights, of the same shape as `a`.  Each value in
        #     `a` only contributes its associated weight towards the bin count
        #     (instead of 1). If `density` is True, the weights are
        #     normalized, so that the integral of the density over the range
        #     remains 1.
        # density : bool, optional
        #     If ``False``, the result will contain the number of samples in
        #     each bin. If ``True``, the result is the value of the
        #     probability *density* function at the bin, normalized such that
        #     the *integral* over the range is 1. Note that the sum of the
        #     histogram values will not be equal to 1 unless bins of unity
        #     width are chosen; it is not a probability *mass* function.
        if transform is None:
            f = self.default_transform
        else:
            f = transform
        hist, bin_edges = np.histogram(
            f(self.y_serie), bins=bins, weights=weights, density=density
        )
        delta = bin_edges[1] - bin_edges[0]

        if cumulative:
            # start = bin_edges[0] - delta
            # hist = np.hstack(([0], np.cumsum(hist) * delta))
            start = bin_edges[0]
            hist = np.cumsum(hist) * delta
        else:
            start = bin_edges[0]

        ret = DSPHistogram(
            name=name,
            samplingStart=start,
            samplingPeriod=delta,
            y_serie=hist,
            default_transform=lambda x: x,
        )
        ret.name_of_x_var = bin_name
        ret.unit_of_x_var = bin_unit
        return ret


class DSPPolarLine(ADSPLine):
    @property
    def dspline_type(self) -> DSPLineType:
        return DSPLineType.POLAR


class DSPNorthPolarLine(ADSPLine):
    @property
    def dspline_type(self) -> DSPLineType:
        return DSPLineType.NORTH_POLAR
