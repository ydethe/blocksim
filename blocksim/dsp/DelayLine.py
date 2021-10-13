import numpy as np
from scipy.interpolate import pchip_interpolate


class DelayLine(object):
    def __init__(self):
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
        xsamp = pchip_interpolate(x=x, xi=self._l_time, yi=self._l_xsamples)
        ysamp = pchip_interpolate(x=x, xi=self._l_time, yi=self._l_ysamples)
        return xsamp + 1j * ysamp
