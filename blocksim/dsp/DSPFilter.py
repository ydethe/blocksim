from abc import abstractmethod

import numpy as np
from numpy import log10, exp, pi, sqrt, cos, sin
from scipy.signal import firwin2, firwin, filtfilt, lfilter, iirdesign

from .. import logger
from ..core.Frame import Frame
from ..core.Node import AComputer, Input, Output, WeightedFIROutput
from .DSPSignal import DSPSignal


__all__ = ["ADSPFilter", "BandpassDSPFilter", "DSPFilter"]


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
        self, name: str, samplingPeriod: float, dtype=np.complex128,
    ):
        AComputer.__init__(self, name=name)

        self.createParameter(name="samplingPeriod", value=samplingPeriod)

        self.defineInput("unfilt", shape=1, dtype=dtype)
        otp = WeightedFIROutput(name="filt", snames=["sample"], dtype=dtype)
        self.addOutput(otp)

    def getTransientPhaseDuration(self) -> float:
        """Returns the duration of the transcient phase of the filter

        Returns:
            Transcient duration (s)

        """
        b, a = self.generateCoefficients()

        if len(a) == 1 and np.abs(a[0] - 1) < 1e-9:
            # FIR filter case
            numtaps = len(b)
            gd = numtaps * self.samplingPeriod / 2
        else:
            # IIR filter case
            gd = 0.0

        return gd

    @abstractmethod
    def generateCoefficients(self):  # pragma: no cover
        """Shall return a and b coefficient of numerator and denominator respectively
        TODO

        Returns:
            A tuple of coefficients (b, a)

        """
        pass

    def compute_outputs(
        self, t1: float, t2: float, unfilt: np.array, filt: np.array,
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
        z = self.process(s.y_serie)

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
        b, a = self.generateCoefficients()

        # z = filtfilt(b, a, s)
        z = lfilter(b, a, s)

        return z


class ArbitraryDSPFilter(ADSPFilter):
    """A filter with custom taps
    https://en.wikipedia.org/wiki/Infinite_impulse_response

    $$ H(z) = B(z)/A(z) = \frac{b_0 + b_1.z^{-1} + \dots + b_{nb-1}.z^{-nb+1}}{a_0 + a_1.z^{-1} + \dots + a_{na-1}.z^{-na+1}} $$

    Args:
        name: Name of the filter
        samplingPeriod: Time spacing of the signal (s)
        btaps: Coefficients of the filter denominator
        ataps: Coefficients of the filter numerator

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        samplingPeriod: float,
        btaps: "array",
        ataps: "array" = np.array([1.0]),
        dtype=np.complex128,
    ):
        ADSPFilter.__init__(self, name=name, samplingPeriod=samplingPeriod, dtype=dtype)

        self.createParameter(name="samplingPeriod", value=samplingPeriod)
        self.createParameter(name="ataps", value=ataps)
        self.createParameter(name="btaps", value=btaps)

    def generateCoefficients(self):
        return self.btaps, self.ataps

    @classmethod
    def fromIIRSpecification(
        cls,
        name: str,
        fs: float,
        wp,
        ws,
        gpass: float,
        gstop: float,
        analog: bool = False,
        ftype: str = "ellip",
    ) -> "ArbitraryDSPFilter":
        """Complete IIR digital and analog filter design.

        Given passband and stopband frequencies and gains, construct an analog or
        digital IIR filter of minimum order for a given basic type.

        Args:
            name: Name of the filter
            fs: The sampling frequency of the digital system (Hz)
            wp, ws (float or array like, shape (2,)):  Passband and stopband edge frequencies in Hz. Possible values are scalars
                (for lowpass and highpass filters) or ranges (for bandpass and bandstop filters).
                For example:

                    - Lowpass:   wp = 20,          ws = 30
                    - Highpass:  wp = 30,          ws = 20
                    - Bandpass:  wp = [20, 50],   ws = [10, 60]
                    - Bandstop:  wp = [10, 60],   ws = [20, 50]

                Note, that for bandpass and bandstop filters passband must lie strictly
                inside stopband or vice versa.
            gpass: The maximum loss in the passband (dB).
            gstop: The minimum attenuation in the stopband (dB).
            analog: When True, return an analog filter, otherwise a digital filter is returned
            ftype: The type of IIR filter to design:

                    - Butterworth   : 'butter'
                    - Chebyshev I   : 'cheby1'
                    - Chebyshev II  : 'cheby2'
                    - Cauer/elliptic: 'ellip'
                    - Bessel/Thomson: 'bessel'

        """
        if analog:
            wp=wp/(2*pi)
            ws=ws/(2*pi)
        b, a = iirdesign(
            wp, ws, gpass, gstop, analog=analog, ftype=ftype, output="ba", fs=fs
        )
        filt = ArbitraryDSPFilter(
            name=name, samplingPeriod=1 / fs, btaps=b, ataps=a, dtype=np.complex128
        )
        return filt

    @classmethod
    def fromFIRSpecification(
        cls,
        name: str,
        fs: float,
        numtaps: int,
        freq,
        gain,
        nfreqs: int = None,
        window: str = "hamming",
        antisymmetric: bool = False,
    ) -> "ArbitraryDSPFilter":
        """
        FIR filter design using the window method.

        From the given frequencies `freq` and corresponding gains `gain`,
        this function constructs an FIR filter with linear phase and
        (approximately) the given frequency response.

        Args:
            name: Name of the filter
            fs: The sampling frequency of the signal. Each frequency in `cutoff` must be between 0 and ``fs/2``.
            numtaps: The number of taps in the FIR filter.  `numtaps` must be less than `nfreqs`.
            freq (array_like, 1-D): The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
                Nyquist.  The Nyquist frequency is half `fs`.
                The values in `freq` must be nondecreasing. A value can be repeated
                once to implement a discontinuity. The first value in `freq` must
                be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must
                not be repeated.
            gain (array_like): The filter gains at the frequency sampling points. Certain
                constraints to gain values, depending on the filter type, are applied,
                see Notes for details.
            nfreqs: The size of the interpolation mesh used to construct the filter.
                For most efficient behavior, this should be a power of 2 plus 1
                (e.g, 129, 257, etc). The default is one more than the smallest
                power of 2 that is not less than `numtaps`. `nfreqs` must be greater
                than `numtaps`.
            window: Window function to use. Default is "hamming". See
                `scipy.signal.get_window` for the complete list of possible values.
                If None, no window function is applied.
            antisymmetric: Whether resulting impulse response is symmetric/antisymmetric.

        """
        b = firwin2(
            numtaps=numtaps,
            freq=freq,
            gain=gain,
            nfreqs=nfreqs,
            window=window,
            antisymmetric=antisymmetric,
            fs=fs,
        )
        filt = ArbitraryDSPFilter(
            name=name,
            samplingPeriod=1 / fs,
            btaps=b,
            ataps=np.array([1.0]),
            dtype=np.complex128,
        )
        return filt


class BandpassDSPFilter(ADSPFilter):
    """A bandpass FIR filter, generated with https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.signal.firwin.html

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
        fs = 1 / self.samplingPeriod

        # https://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need/31077
        d = 10e-2
        nt = int(-2 / 3 * log10(10 * d ** 2) * fs / (self.f_high - self.f_low))
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

        return b, np.array([1.0])
