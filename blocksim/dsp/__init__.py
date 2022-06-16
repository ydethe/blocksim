"""This module provides signal processing functions
BOC and PSK modulator are available, as well as antenna network, Klobuchar model, delay lines and more.

"""

from math import factorial
from itertools import product
from typing import List, Union, Any

from tqdm import tqdm
from nptyping import NDArray, Shape
import numpy as np
from numpy import sqrt, sign, pi, exp
from numpy.fft import fft, ifft
from scipy import linalg as lin


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


def analyse_DV(
    wavelength: float,
    period: float,
    dist0: float,
    damb: float,
    vrad0: float,
    vamb: float,
    seq: "blocksim.dsp.DSPSignal.DSPSignal",
    rxsig: "blocksim.dsp.DSPSignal.DSPSignal",
    nv: int,
    n_integration: int = -1,
    coherent: bool = True,
    progress_bar: bool = False,
    corr_window: str = "hamming",
) -> "blocksim.dsp.DSPMap.DSPRectilinearMap":
    """Distance / velocity analysis for acquisition

    Args:
        wavelength: Wavelength of the carrier (m)
        period: Window length (s)
        dist0: Center of the distance research domain (m)
        damb: Width of the distance research domain (m)
        vrad0: Center of the velocity research domain (m/s)
        vamb: Width of the velocity research domain (m/s)
        seq: Local replica of the signal
        rxsig: Received signal to be analysed
        nv: Number of velocity hypothesis to be tested
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

    if nv % 2 == 0:
        nv += 1

    dt = rxsig.samplingPeriod

    if period < damb / c:
        disp_p, _, lbl_p, unit_p = getUnitAbbrev(samp=period, unit="s")
        disp_d, _, lbl_d, unit_d = getUnitAbbrev(samp=damb / c, unit="s")
        raise AssertionError(
            f"Period window is shorter than distance window ({disp_p:.2f} {lbl_p}{unit_p} vs {disp_d:.2f} {lbl_d}{unit_d})"
        )

    # Number of sample in a period
    nb_samples_in_period = int(period / dt)

    # Number of samples in a distance window
    nb_samples_in_damb = int(damb / c / dt)

    # Sample index of the center distance in the first window
    n0 = int(dist0 / c / dt)

    kmin = n0 - nb_samples_in_damb // 2
    kmax = kmin + nb_samples_in_damb
    img = np.empty((nb_samples_in_damb, nv), dtype=np.complex128)

    tab_v = np.linspace(vrad0 - vamb / 2, vrad0 + vamb / 2, nv)
    if progress_bar:
        v_gen = tqdm(tab_v)
    else:
        v_gen = tab_v

    for kv, vrad in enumerate(v_gen):
        fd = -vrad / wavelength

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

        img[:, kv] = zi.y_serie[kmin:kmax]

    spg = DSPRectilinearMap(
        name="spg",
        samplingXStart=tab_v[0] - vrad0,
        samplingXPeriod=tab_v[1] - tab_v[0],
        samplingYStart=zi.samplingStart,
        samplingYPeriod=zi.samplingPeriod,
        img=img,
        default_transform=np.abs,
    )
    spg.name_of_x_var = "Radial Velocity (%.1f m/s delta)" % vrad0
    spg.unit_of_x_var = "m/s"
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


def createGoldSequence(
    name: str,
    sv: Union[List[int], int],
    repeat=1,
    chip_rate: float = 1.023e6,
    sampling_factor: int = 10,
    samplingStart: float = 0,
    bitmap=[-1, 1],
) -> "blocksim.dsp.DSPSignal.DSPSignal":
    """Builds Gold sequence

    Args:
        name: Name of the signal
        sv: Identifier of the SV. Can be either the PRN number (int), or the code tap selection (list of 2 int)
        repeat: Number of copies of a 1023 Gold sequence
        chip_rate: Sampling frequency of the signal (Hz)
        sampling_factor: Factor so that fs = sampling_factor*chip_rate
        samplingStart: First date of the sample of the signal (s)
        bitmap: List of 2 values to map the bits on. [0, 1] returns a sequence with 0 and 1

    Returns:
        The DSPSignal. All the samples are in the given bitmap

    """
    from .DSPSignal import DSPSignal

    SV_list = {
        1: [2, 6],
        2: [3, 7],
        3: [4, 8],
        4: [5, 9],
        5: [1, 9],
        6: [2, 10],
        7: [1, 8],
        8: [2, 9],
        9: [3, 10],
        10: [2, 3],
        11: [3, 4],
        12: [5, 6],
        13: [6, 7],
        14: [7, 8],
        15: [8, 9],
        16: [9, 10],
        17: [1, 4],
        18: [2, 5],
        19: [3, 6],
        20: [4, 7],
        21: [5, 8],
        22: [6, 9],
        23: [1, 3],
        24: [4, 6],
        25: [5, 7],
        26: [6, 8],
        27: [7, 9],
        28: [8, 10],
        29: [1, 6],
        30: [2, 7],
        31: [3, 8],
        32: [4, 9],
    }

    if not hasattr(sv, "__iter__"):
        sv = SV_list[sv]

    # init registers
    G1 = [1 for _ in range(10)]
    G2 = [1 for _ in range(10)]

    ca = []
    for _ in range(1023):
        g1 = shift(G1, [3, 10], [10])  # feedback 3,10, output 10
        g2 = shift(
            G2, [2, 3, 6, 8, 9, 10], sv
        )  # feedback 2,3,6,8,9,10, output sv for sat
        ca.extend([(g1 + g2) % 2] * sampling_factor)

    bits = np.array(ca * repeat, dtype=np.int8)
    a, b = bitmap
    seq = (b - a) * bits + a
    sig = DSPSignal(
        name=name,
        samplingStart=samplingStart,
        samplingPeriod=1 / chip_rate / sampling_factor,
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
    sig = DSPSignal(
        name=name, samplingStart=0, samplingPeriod=1 / sampling_freq, y_serie=seq
    )
    return sig
