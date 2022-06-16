from abc import abstractmethod
from typing import Tuple, Any

from nptyping import NDArray, Shape
import numpy as np
from numpy import log10, log2
from scipy import linalg as lin
from scipy.signal import (
    firwin2,
    firwin,
    lfilter,
    iirdesign,
    TransferFunction,
    remez,
    firls,
    freqz,
)

from .. import logger
from ..dsp import phase_unfold
from ..core.Node import AComputer, TFOutput
from .DSPSignal import DSPSignal
from .DSPLine import ADSPLine
from ..graphics.GraphicSpec import DSPLineType


__all__ = ["DSPBodeDiagram", "ADSPFilter", "BandpassDSPFilter", "ArbitraryDSPFilter"]


class DSPBodeDiagram(ADSPLine):
    @property
    def dspline_type(self) -> DSPLineType:
        return DSPLineType.BODE_DIAG


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
        otp = TFOutput(name="filt", snames=["sample"], dtype=dtype)
        self.addOutput(otp)

    def getGroupDelay(self) -> float:
        """Returns the group delay of the filter

        Returns:
            Group delay (s)

        """
        b, a = self.generateCoefficients()

        if len(a) == len(b) and np.abs(a[0] - 1) < 1e-9 and lin.norm(a[1:]) < 1e-9:
            # FIR filter case
            numtaps = len(b)
            gd = numtaps * self.samplingPeriod / 2
        else:
            # IIR filter case
            # https://www.dsprelated.com/showarticle/69.php
            # https://github.com/spatialaudio/group-delay-of-filters/blob/main/examples/digital-iir-filters.py
            # group_delayz(b, a, w, fs=1/self.samplingPeriod)
            gd = 0.0

        return gd

    @abstractmethod
    def generateCoefficients(self) -> Tuple["array", "array"]:  # pragma: no cover
        r"""Shall return a and b coefficient of numerator and denominator respectively
        Coefficients for both the numerator and denominator should be specified in descending exponent order
        (e.g. \( z^2 + 3.z + 5 \) would be represented as [1, 3, 5]).

        $$ H(z) = B(z)/A(z) = \frac{b_0.z^{nb-1} + b_1.z^{nb-2} + \dots + b_{nb-1}}{a_0.z^{na-1} + a_1.z^{na-2} + \dots + a_{na-1}} $$

        Returns:
            A tuple of coefficients (b, a)

        """
        pass

    def update(
        self,
        t1: float,
        t2: float,
        unfilt: NDArray[Any, Any],
        filt: NDArray[Any, Any],
    ) -> dict:
        assert len(unfilt) == 1

        # The actual filtering is made in TFOutput
        outputs = {}
        outputs["filt"] = unfilt

        return outputs

    def apply(self, s: DSPSignal) -> DSPSignal:
        """Filters a DSPSignal without having to create a `blocksim.Simulation.Simulation`

        Args:
            s: The DSPSignal to filter

        Returns:
            The filtered signal

        """
        z = self.process(s.y_serie)

        return DSPSignal(
            name="Filtered %s" % s.name,
            samplingStart=s.samplingStart - self.getGroupDelay(),
            samplingPeriod=s.samplingPeriod,
            y_serie=z,
            default_transform=np.real,
        )

    def process(self, s: NDArray[Any, Any]) -> NDArray[Any, Any]:
        """Filters a signal without having to create a `blocksim.Simulation.Simulation`

        Args:
            s: The array of (complex) samples to filter

        Returns:
            The filtered signal

        """
        b, a = self.generateCoefficients()

        z = lfilter(b, a, s)

        return z

    def bodeDiagram(
        self,
        name: str,
        fpoints: int = 200,
    ) -> DSPBodeDiagram:
        """Computes the Bode diagram for the filter

        Args:
            name: Name of the diagram
            fpoints: If int, number of frequency samples to use for the plot
                     If iterable, list of frequency samples to use for the plot

        Returns:
            The Bode diagram

        """
        fs = 1 / self.samplingPeriod

        b, a = self.generateCoefficients()

        if hasattr(fpoints, "__iter__"):
            freq = fpoints
        else:
            freq = np.arange(0, fs / 2, fs / 2 / fpoints)

        num, den = TransferFunction._z_to_zinv(b, a)
        _, y = freqz(num, den, worN=freq, fs=fs)

        ret = DSPBodeDiagram(
            name=name,
            samplingStart=freq[0],
            samplingPeriod=freq[1] - freq[0],
            y_serie=y,
            default_transform=ADSPLine.to_db_lim(-100),
        )
        ret.unit_of_x_var = "Hz"
        ret.name_of_x_var = "Frequency"

        return ret


class ArbitraryDSPFilter(ADSPFilter):
    r"""A filter with custom taps
    https://en.wikipedia.org/wiki/Infinite_impulse_response
    Coefficients for both the numerator and denominator should be specified in descending exponent order
    (e.g. \( z^2 + 3.z + 5 \) would be represented as [1, 3, 5]).

    $$ H(z) = B(z)/A(z) = \frac{b_0.z^{nb-1} + b_1.z^{nb-2} + \dots + b_{nb-1}}{a_0.z^{na-1} + a_1.z^{na-2} + \dots + a_{na-1}} $$

    Args:
        name: Name of the filter
        samplingPeriod: Time spacing of the signal (s)
        num: Coefficients of the filter denominator
        den: Coefficients of the filter numerator

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        samplingPeriod: float,
        num: NDArray[Any, Any],
        den: NDArray[Any, Any] = None,
        dtype=np.complex128,
    ):
        ADSPFilter.__init__(self, name=name, samplingPeriod=samplingPeriod, dtype=dtype)

        if den is None:
            den = np.zeros_like(num)
            den[0] = 1

        self.createParameter(name="samplingPeriod", value=samplingPeriod)
        self.createParameter(name="den", value=np.array(den.copy()))
        self.createParameter(name="num", value=np.array(num.copy()))

    def generateCoefficients(self):
        return self.num.copy(), self.den.copy()

    def to_dlti(self) -> TransferFunction:
        """Creates a scipy TransferFunction instance.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html

        Returns:
            The TransferFunction instance

        """
        b, a = self.generateCoefficients()
        sys = TransferFunction(b, a, dt=self.samplingPeriod)
        return sys

    @classmethod
    def fromIIRSpecification(
        cls,
        name: str,
        fs: float,
        wp,
        ws,
        gpass: float,
        gstop: float,
        ftype: str = "ellip",
    ) -> "blocksim.dsp.DSPFilter.ArbitraryDSPFilter":
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
            ftype: The type of IIR filter to design:

                    - Butterworth   : 'butter'
                    - Chebyshev I   : 'cheby1'
                    - Chebyshev II  : 'cheby2'
                    - Cauer/elliptic: 'ellip'
                    - Bessel/Thomson: 'bessel'

        """
        b, a = iirdesign(
            wp, ws, gpass, gstop, analog=False, ftype=ftype, output="ba", fs=fs
        )
        filt = ArbitraryDSPFilter(
            name=name,
            samplingPeriod=1 / fs,
            num=b,
            den=a,
            dtype=np.complex128,
        )
        return filt

    @classmethod
    def fromFIRSpecification(
        cls,
        name: str,
        fs: float,
        numtaps: int,
        bands: NDArray[Any, Any],
        desired: NDArray[Any, Any],
        method: str = "firwin2",
        **kwargs,
    ) -> "blocksim.dsp.DSPFilter.ArbitraryDSPFilter":
        """
        FIR filter design using the window method.

        From the given frequencies *freq* and corresponding gains *gain*,
        this function constructs an FIR filter with linear phase and
        (approximately) the given frequency response.

        Args:
            name: Name of the filter
            fs: The sampling frequency of the signal (Hz)
            numtaps: The number of taps in the FIR filter
            bands (array_like): A monotonic sequence containing the band edges (Hz)
                All elements must be non-negative and less than fs/2
            desired (array_like): A sequence half the size of bands containing the desired gain
                in each of the specified bands
            method: One of 'firwin2', 'remez', 'ls'

        """
        if method == "firwin2":
            nfreqs = 1 + 2 ** int(np.ceil(log2(numtaps)))
            nbands = len(desired)
            n_per_band = nfreqs // nbands

            if bands[0] == 0:
                freq_a = []
                gain_a = []
            else:
                freq_a = [0]
                gain_a = [0]

            for k in range(nbands):
                f1, f2 = bands[2 * k : 2 * k + 2]
                bf = np.linspace(f1, f2, n_per_band)
                freq_a.extend(bf)
                gain_a.extend(desired[k] * np.ones_like(bf))

            if bands[-1] != fs / 2:
                freq_a.append(fs / 2)
                gain_a.append(0)

            freq = kwargs.pop("freq", freq_a)
            gain = kwargs.pop("gain", gain_a)

            if numtaps % 2 == 0 and gain[0] == 0 and gain[-1] == 0.0:
                # Filter type II or IV
                antisymmetric = True  # Could be False if we want to
            elif numtaps % 2 == 0 and gain[0] == 0 and gain[-1] != 0.0:
                # Filter type IV
                antisymmetric = False
            elif numtaps % 2 == 0 and gain[0] != 0 and gain[-1] == 0.0:
                # Filter type II
                antisymmetric = True
            elif numtaps % 2 == 0 and gain[0] != 0 and gain[-1] != 0.0:
                # Filter type
                raise ValueError(f"N={numtaps},H(1)={gain[0]},H(-1)={gain[-1]}")
            elif numtaps % 2 == 1 and gain[0] == 0 and gain[-1] == 0.0:
                # Filter type I or III
                antisymmetric = True  # Could be False if we want to
            elif numtaps % 2 == 1 and gain[0] == 0 and gain[-1] != 0.0:
                # Filter type I
                antisymmetric = True
            elif numtaps % 2 == 1 and gain[0] != 0 and gain[-1] == 0.0:
                # Filter type I
                antisymmetric = True
            elif numtaps % 2 == 1 and gain[0] != 0 and gain[-1] != 0.0:
                # Filter type I
                antisymmetric = True

            # kwargs: window='hamming'
            b = firwin2(
                numtaps=numtaps,
                freq=freq,
                gain=gain,
                antisymmetric=antisymmetric,
                fs=fs,
                **kwargs,
            )
        elif method == "remez":
            # kwargs: weight=None, type='bandpass', maxiter=25, grid_density=16
            b = remez(numtaps, bands, desired, fs=fs, **kwargs)
        elif method == "ls":
            # kwargs: weight=None
            desired_ls = np.empty_like(bands)
            desired_ls[::2] = desired
            desired_ls[1::2] = desired
            b = firls(numtaps, bands, desired_ls, fs=fs, **kwargs)

        a = np.zeros_like(b)
        a[0] = 1

        filt = ArbitraryDSPFilter(
            name=name,
            samplingPeriod=1 / fs,
            num=b,
            den=a,
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

    def generateCoefficients(self) -> NDArray[Any, Any]:
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

        a = np.zeros_like(b)
        a[0] = 1
        return b, a
