"""This module provides signal processing functions
BOC and PSK modulator are available, as well as antenna network, Klobuchar model, delay lines and more.

"""

from math import factorial
from itertools import product
from typing import List, Union, Any

import rich.progress as rp
from nptyping import NDArray, Shape
import numpy as np
from numpy import sqrt, sign, pi, exp
from numpy.fft import fft, ifft
from scipy import linalg as lin
from scipy.interpolate import interp1d

from ..gnss.GNSScodegen import GNSScodegen


def derivative_coeff(rank: int = 1, order: int = None) -> NDArray[Any, Any]:
    """Computes the coefficients of a derivaive estimator

    Args:
        rank: Rank of the derivative
        order: Order of the Taylor serie used to estimate the derivative. Shall be >= rank
            The default value is *rank*

    Returns:
        Taps of a derivating filter

    """
    if order is None:
        order = rank

    if order < rank:
        raise AssertionError("order=%s, rank=%s" % (order, rank))

    k = int(np.ceil(order / 2))

    # Dans wxMaxima:
    # k:3;A:genmatrix(lambda([i, j],(if (i # 0 or j+k#0) then (i^(j+k)/factorial(j+k)) else 1)), k,k,-k,-k);
    # k:2$;invert(genmatrix(lambda([i, j],(if (i # 0 or j+k#0) then (i^(j+k)/factorial(j+k)) else 1)), k,k,-k,-k));
    n = 2 * k + 1
    A = np.empty((n, n))
    for r, s in product(range(n), repeat=2):
        A[r, s] = (r - k) ** s / factorial(s)

    y = np.zeros(n)
    y[rank] = 1
    coeffs = lin.solve(a=A, b=y, overwrite_a=True, overwrite_b=True, transposed=True)

    return coeffs


def get_window(win, n: int) -> NDArray[Any, Any]:
    """Creates the samples of a window.
    The resulting window guarantees that a CW with amplitude A will keep its amplitude after windowing

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html for the list of available windows

    Args:
        win: The window to use
        n: Number of samples

    Returns:
        The array of coefficients

    """
    from scipy.signal import get_window

    w = get_window(win, n)
    nrm = np.sum(w) / n

    return w / nrm


def phase_unfold(sig: NDArray[Any, Any], eps: float = 1e-9) -> NDArray[Any, Any]:
    """Unfolds the phase law of the given complex signal

    Args:
        sig: The array of complex samples
        eps: The threshold to test equality

    Returns:
        The unfolded phase law (rad)

    """
    n = len(sig)
    pha = np.zeros(n)
    pha[0] = np.angle(sig[0])

    init_ok = False
    i = 0
    while True:
        if np.abs(sig[i]) > eps:
            init_ok = True
            pha[0 : 1 + i] = np.angle(sig[i])
            break

        i += 1
        if i == n:
            break

    if not init_ok or i == n - 1:
        return pha

    for j in range(i + 1, n):
        if np.abs(sig[j - 1]) < eps or np.abs(sig[j]) < eps:
            r = 0
        else:
            r = sig[j] / sig[j - 1]
            r /= np.abs(r)

        # Nyquist–Shannon sampling theorem garantees that |dpha| < pi
        # So we can call np.angle which will not produce any ambiguity
        dpha = np.angle(r)

        pha[j] = pha[j - 1] + dpha

    return pha


def phase_unfold_deg(sig: NDArray[Any, Any], eps: float = 1e-9) -> NDArray[Any, Any]:
    """Unfolds the phase law of the given complex signal

    Args:
        sig: The array of complex samples
        eps: The threshold to test equality

    Returns:
        The unfolded phase law (deg)

    """
    return 180 / pi * phase_unfold(sig, eps=eps)


def delay_doppler_analysis(
    period: float,
    delay_search_center: float,
    delay_search_win: float,
    doppler_search_center: float,
    doppler_search_win: float,
    seq: "blocksim.dsp.DSPSignal.DSPSignal",
    rxsig: "blocksim.dsp.DSPSignal.DSPSignal",
    ndop: int,
    n_integration: int = -1,
    coherent: bool = True,
    progress_bar: bool = False,
    corr_window: str = "hamming",
) -> "blocksim.dsp.DSPMap.DSPRectilinearMap":
    """Delay / doppler analysis for acquisition

    Args:
        period: Window length (s)
        delay_search_center: Center of the delay research domain (s)
        delay_search_win: Width of the delay research domain (s)
        doppler_search_center: Center of the doppler research domain (Hz)
        doppler_search_win: Width of the velocity research domain (m/s)
        seq: Local replica of the signal
        rxsig: Received signal to be analysed
        ndop: Number of doppler hypothesis to be tested
        n_integration: Number of period to sum. A value of -1 means to sum everything
        coherent: To use coherent integration. Non coherent otherwise
        progress_bar: To turn on the display of a progress bar
        corr_window: Window to be used for correlation

    Returns:
        The DSPRectilinearMap of the analysis

    """
    from ..constants import c
    from .DSPMap import DSPRectilinearMap
    from ..graphics import getUnitAbbrev

    if ndop % 2 == 0:
        ndop += 1

    dt = rxsig.samplingPeriod

    if period < delay_search_win:
        disp_p, _, lbl_p, unit_p = getUnitAbbrev(samp=period, unit="s")
        disp_d, _, lbl_d, unit_d = getUnitAbbrev(samp=delay_search_win, unit="s")
        raise AssertionError(
            f"Period window is shorter than delay window ({disp_p:.2f} {lbl_p}{unit_p} vs {disp_d:.2f} {lbl_d}{unit_d})"
        )

    # Number of sample in a period
    nb_samples_in_period = int(period / dt)

    # Number of samples in a delay window
    nb_samples_in_delay_win = int(delay_search_win / dt)

    # Sample index of the center delay in the first window
    n0 = int(delay_search_center / dt)

    kmin = n0 - nb_samples_in_delay_win // 2
    kmax = kmin + nb_samples_in_delay_win

    if kmax > nb_samples_in_period:
        raise AssertionError(f"Doppler window larger than period")

    img = np.empty((nb_samples_in_delay_win, ndop), dtype=np.complex128)

    tab_dop = np.linspace(
        doppler_search_center - doppler_search_win / 2,
        doppler_search_center + doppler_search_win / 2,
        ndop,
    )
    if progress_bar:
        dop_gen = rp.track(tab_dop, description="DV analyze...")
    else:
        dop_gen = tab_dop

    for kd, fd in enumerate(dop_gen):
        # Doppler compensation
        dop_free = rxsig.applyDopplerFrequency(fdop=-fd)

        # Correlation
        corr = dop_free.correlate(seq, win=corr_window)

        # Integration
        zi = corr.integrate(
            period=period,
            n_integration=n_integration,
            offset=-corr.samplingStart,
            coherent=coherent,
        )

        img[:, kd] = zi.y_serie[kmin:kmax]

    spg = DSPRectilinearMap(
        name="spg",
        samplingXStart=tab_dop[0],
        samplingXPeriod=tab_dop[1] - tab_dop[0],
        samplingYStart=zi.samplingStart,
        samplingYPeriod=zi.samplingPeriod,
        img=img,
        default_transform=np.abs,
    )

    spg.name_of_x_var = "Doppler"
    spg.unit_of_x_var = "Hz"
    spg.name_of_y_var = "Delay"
    spg.unit_of_y_var = "s"

    return spg


def shift(register: list, feedback: list, output: list):
    """GPS Shift Register

    Args:
        feedback: which positions to use as feedback (0 indexed)
        output: which positions are output (0 indexed)

    Returns
        Output of shift register

    Examples:
        >>> G1 = [1,1,1,1,1,1,1,1,1,1]
        >>> out = shift(G1, [3,10], [10])
        >>> G1
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        >>> out
        1

    """
    # calculate output
    out = [register[i - 1] for i in output]
    if len(out) > 1:
        out = sum(out) % 2
    else:
        out = out[0]

    # modulo 2 add feedback
    fb = sum([register[i - 1] for i in feedback]) % 2

    # shift to the right
    for i in reversed(range(len(register[1:]))):
        register[i + 1] = register[i]

    # put feedback in position 1
    register[0] = fb

    return out


def createGNSSSequence(
    name: str,
    modulation: str,
    sv: int,
    repeat=1,
    chip_rate: float = 1.023e6,
    samplingStart: float = 0,
    bitmap=[-1, 1],
    samples_per_chip: int = 1,
) -> "blocksim.dsp.DSPSignal.DSPSignal":
    """Builds Gold sequence

    Args:
        name: Name of the signal
        modulation:

        * GPS: L1CA, L2CM, L2CL, L5I, L5Q.
        * Galileo: E1B, E1C, E5aI, E5aQ, E5bI, E5bQ, E6-B, E6-C.
        * BeiDou: B1I.

        sv: Identifier of the SV
        repeat: Number of copies of a 1023 Gold sequence
        chip_rate: Chip rate (Hz)
        samplingStart: First date of the sample of the signal (s)
        bitmap: List of 2 values to map the bits on. [0, 1] returns a sequence with 0 and 1
        samples_per_chip: Number of samples per chip

    Returns:
        The DSPSignal. All the samples are in the given bitmap

    """
    from .DSPSignal import DSPSignal

    ca = GNSScodegen(sv, modulation)
    ca = np.repeat(ca, samples_per_chip)

    bits = np.tile(ca, reps=repeat)
    a, b = bitmap
    seq = (b - a) / 2 * bits + (b + a) / 2
    sig = DSPSignal(
        name=name,
        samplingStart=samplingStart,
        samplingPeriod=1 / chip_rate / samples_per_chip,
        y_serie=seq,
        dtype=np.int64,
    )
    return sig


def zadoff_chu(u, n):
    """Create a Zadoff-Chu sequence.

    See https://en.wikipedia.org/wiki/Zadoff-Chu_sequence

    Args:
        u: Root of the sequence
        n: Length of the sequence

    Returns:
        The array of complex samples

    """
    k = np.arange(n)
    return exp(-1j * pi * u * k * (k + 1) / n)


def createZadoffChu(
    name: str,
    n_zc: int,
    u: int,
    sampling_freq: float,
    samplingStart: float = 0,
) -> "blocksim.dsp.DSPSignal.DSPSignal":
    """Builds DSPSignal defined by a Zadoff-Chu sequence

    Args:
        name: Name of the signal
        n_zc: Length of the Zadoff-Chu sequence
        u: Index of the Zadoff-Chu sequence

    Returns:
        The DSPSignal

    """
    from .DSPSignal import DSPSignal

    seq = zadoff_chu(u, n_zc)
    sig = DSPSignal(name=name, samplingStart=0, samplingPeriod=1 / sampling_freq, y_serie=seq)
    return sig
