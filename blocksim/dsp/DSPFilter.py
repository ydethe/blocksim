from collections import deque

import numpy as np
from numpy import log10, exp, pi, sqrt, cos, sin
from scipy.signal import firwin2, firwin, lfilter_zi, lfilter

from ..core.Frame import Frame
from ..core.Node import AComputer, Input, Output
from .DSPSignal import DSPSignal


__all__ = ["CircularBuffer", "DSPFilter"]


class CircularBuffer(deque):
    def __init__(self, size=0):
        super(CircularBuffer, self).__init__(maxlen=size)
        for _ in range(size):
            self.append(0.0)


class WeightedOutput(Output):

    __slots__ = ["__taps", "__buf"]

    def __init__(self, name: str, dtype):
        Output.__init__(self, name=name, snames=["sample"], dtype=dtype)
        self.setInitialState(initial_state=np.array([0], dtype=dtype))

    def resetCallback(self, frame: Frame):
        filt = self.getComputer()

        self.__taps = filt.generateCoefficients()
        n = len(self.__taps)
        self.__buf = CircularBuffer(size=n)

    def getTaps(self) -> np.array:
        return self.__taps

    def processSample(self, sample: np.complex128) -> np.complex128:
        self.__buf.append(sample)
        buf = np.array(self.__buf, dtype=self.getDataType())
        res = buf @ self.__taps
        return res


class DSPFilter(AComputer):
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

    __slots__ = []

    def __init__(
        self,
        name: str,
        f_low: float,
        f_high: float,
        numtaps: int,
        samplingPeriod: float,
        win: str = "hamming",
        dtype=np.complex128,
    ):
        AComputer.__init__(self, name=name)

        self.createParameter(name="f_low", value=f_low)
        self.createParameter(name="f_high", value=f_high)
        self.createParameter(name="numtaps", value=numtaps)
        self.createParameter(name="win", value=win)
        self.createParameter(name="samplingPeriod", value=samplingPeriod)

        self.defineInput("unfilt", shape=1, dtype=dtype)
        otp = WeightedOutput(name="filt", dtype=dtype)
        self.addOutput(otp)

    def generateCoefficients(self) -> np.array:
        """Generates the filter's coefficients

        Returns:
          The coefficients

        """
        fs = 1 / self.samplingPeriod

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

    def getTransientPhaseDuration(self) -> float:
        return self.numtaps * self.samplingPeriod / 2

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        unfilt: np.array,
        filt: np.array,
    ) -> dict:
        otp = self.getOutputByName("filt")

        assert len(unfilt) == 1
        filt = otp.processSample(unfilt[0])
        outputs = {}
        outputs["filt"] = np.array([filt])

        return outputs

    def apply(self, s: DSPSignal) -> DSPSignal:
        """Filters a :class:`blocksim.dsp.DSPSignal.DSPSignal` without having to create a blocksim.Simulation.Simulation

        Args:
          s
            The :class:`blocksim.dsp.DSPSignal.DSPSignal` to filter

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
