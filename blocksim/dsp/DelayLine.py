from abc import ABCMeta, abstractmethod

import numpy as np

from .CircularBuffer import CircularBuffer

from .. import logger


class ADelayLine(metaclass=ABCMeta):

    __slots__ = []

    @abstractmethod
    def addSample(self, t: float, sample: np.complex128):
        pass

    @abstractmethod
    def getDelayedSample(self, delay: float) -> np.complex128:
        pass


class InfiniteDelayLine(ADelayLine):

    __slots__ = ["_l_time", "_l_xsamples", "_l_ysamples"]

    def __init__(self):
        ADelayLine.__init__(self)

        self._l_time = []
        self._l_xsamples = []
        self._l_ysamples = []

    def addSample(self, t: float, sample: np.complex128):
        self._l_time.append(t)
        self._l_xsamples.append(np.real(sample))
        self._l_ysamples.append(np.imag(sample))

    def getDelayedSample(self, delay: float) -> np.complex128:
        x = self._l_time[-1] - delay

        if x < 0 or len(self._l_time) < 2:
            return 0

        xsamp = np.interp(x=x, xp=self._l_time, fp=self._l_xsamples, left=0, right=0)
        ysamp = np.interp(x=x, xp=self._l_time, fp=self._l_ysamples, left=0, right=0)
        # xsamp = pchip_interpolate(x=x, xi=self._l_time, yi=self._l_xsamples)
        # ysamp = pchip_interpolate(x=x, xi=self._l_time, yi=self._l_ysamples)

        return xsamp + 1j * ysamp


class FiniteDelayLine(ADelayLine):

    __slots__ = ["_l_time", "_l_samples", "__size"]

    def __init__(self, size: int, dtype):
        ADelayLine.__init__(self)

        self.__size = size
        self._l_time = CircularBuffer(size=size, dtype=np.float64, fill_with=np.nan)
        self._l_samples = CircularBuffer(size=size, dtype=dtype, fill_with=np.nan)

    def addSample(self, t: float, sample: np.complex128):
        self._l_time.append(t)
        self._l_samples.append(sample)

        if not np.isnan(self._l_time[1]):
            self._l_time.doubleBufferSize()
            self._l_samples.doubleBufferSize()

    def getDelayedSample(self, delay: float) -> np.complex128:
        x = self._l_time[-1] - delay

        itime = self._l_time.search(x)

        if np.isnan(itime) or itime < 0:
            return 0

        ta = self._l_time[itime]
        tb = self._l_time[itime + 1]
        za = self._l_samples[itime]
        zb = self._l_samples[itime + 1]

        if np.isnan(ta) or ta > x:
            raise AssertionError(
                "Violation of constraint ta <= x : ta=%g, x=%g" % (ta, x)
            )

        if np.isnan(tb) or x > tb:
            raise AssertionError(
                "Violation of constraint x < tb : x=%g, tb=%g" % (x, tb)
            )

        samp = (zb - za) / (tb - ta) * (x - ta) + za

        return samp
