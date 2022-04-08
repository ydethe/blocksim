from abc import abstractmethod

import numpy as np
from numpy import log10, exp, pi, sqrt, cos, sin
from scipy.signal import firwin2, firwin, lfilter_zi, lfilter

from .. import logger
from ..core.Frame import Frame
from ..core.Node import AComputer, Input, Output
from .DSPSignal import DSPSignal
from .CircularBuffer import CircularBuffer


__all__ = ["ADSPFilter", "BandpassDSPFilter", "DSPFilter"]


class WeightedOutput(Output):

    __slots__ = ["__taps", "__buf"]

    def __init__(self, name: str, dtype):
        Output.__init__(self, name=name, snames=["sample"], dtype=dtype)
        self.setInitialState(initial_state=np.array([0], dtype=dtype))

    def resetCallback(self, frame: Frame):
        filt = self.getComputer()
        typ = self.getDataType()

        self.__taps = filt.generateCoefficients()
        n = len(self.__taps)
        self.__buf = CircularBuffer(size=n, dtype=typ)

    def processSample(self, sample: np.complex128) -> np.complex128:
        self.__buf.append(sample)
        buf = self.__buf.getAsArray()
        res = buf @ self.__taps
        return res


class ADSPFilter(AComputer):
    """A generic abstract filter

    Attributes:
        samplingPeriod: the sampling period (s)

    Args:
        name: Name of the spectrum
        samplingPeriod: Time spacing of the signal (s)

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        samplingPeriod: float,
        dtype=np.complex128,
    ):
        AComputer.__init__(self, name=name)

        self.createParameter(name="samplingPeriod", value=samplingPeriod)

        self.defineInput("unfilt", shape=1, dtype=dtype)
        otp = WeightedOutput(name="filt", dtype=dtype)
        self.addOutput(otp)

    def getTransientPhaseDuration(self) -> float:
        """Returns the duration of the transcient phase of the filter

        Returns:
            Transcient duration (s)

        """
        numtaps = len(self.generateCoefficients())
        return numtaps * self.samplingPeriod / 2

    @abstractmethod
    def generateCoefficients(self):  # pragma: no cover
        pass

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
        """Filters a DSPSignal without having to create a blocksim.Simulation.Simulation

        Args:
            s: The DSPSignal to filter

        Returns:
            The filtered signal

        """
        b = self.generateCoefficients()

        zi = lfilter_zi(b, [1])
        z, _ = lfilter(b, [1], s.y_serie, zi=zi * s.y_serie[0])

        return DSPSignal(
            name="Filtered %s" % s.name,
            samplingStart=s.samplingStart - self.getTransientPhaseDuration(),
            samplingPeriod=s.samplingPeriod,
            y_serie=z,
            default_transform=np.real,
        )

    def process(self, s: "array") -> "array":
        """Filters a signal without having to create a blocksim.Simulation.Simulation

        Args:
            s: The signal to filter

        Returns:
            The filtered signal

        """
        b = self.generateCoefficients()

        zi = lfilter_zi(b, [1])
        z, _ = lfilter(b, [1], s, zi=zi * s[0])

        return z


class ArbitraryDSPFilter(ADSPFilter):
    """A filter with custom taps

    Args:
        name: Name of the filter
        samplingPeriod: Time spacing of the signal (s)
        taps: Coefficients of the filter

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        samplingPeriod: float,
        taps: "array",
        dtype=np.complex128,
    ):
        ADSPFilter.__init__(self, name=name, samplingPeriod=samplingPeriod, dtype=dtype)

        self.createParameter(name="samplingPeriod", value=samplingPeriod)
        self.createParameter(name="taps", value=taps)

    def generateCoefficients(self):
        return self.taps


class BandpassDSPFilter(ADSPFilter):
    """A bandpass filter, generated with https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.signal.firwin.html

    Args:
        name: Name of the spectrum
        f_low: Start frequency of the band pass (Hz)
        f_high: End frequency of the band pass (Hz)
        numtaps: Number of coefficients
        win: The window to be applied. See `blocksim.dsp.get_window`

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
        ADSPFilter.__init__(self, name=name, samplingPeriod=samplingPeriod, dtype=dtype)

        self.createParameter(name="f_low", value=f_low)
        self.createParameter(name="f_high", value=f_high)
        self.createParameter(name="numtaps", value=numtaps)
        self.createParameter(name="win", value=win)

    def generateCoefficients(self) -> "array":
        """Generates the filter's coefficients

        Returns:
          The coefficients

        """
        fs = 1 / self.samplingPeriod

        # https://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need/31077
        d = 10e-2
        nt = int(-2 / 3 * log10(10 * d**2) * fs / (self.f_high - self.f_low))
        if nt > self.numtaps:
            raise ValueError(self.numtaps, nt)

        if self.f_low == 0:
            co = self.f_high
            pz = True
        else:
            co = [self.f_low, self.f_high]
            pz = False
        # logger.debug("%s, %f"%(str(co),1/self.samplingPeriod))

        b = firwin(
            numtaps=self.numtaps,
            cutoff=co,
            pass_zero=pz,
            scale=True,
            window=self.win,
            fs=fs,
        )

        return b
