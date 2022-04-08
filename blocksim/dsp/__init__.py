from math import factorial
from itertools import product
from typing import List, Union

from tqdm import tqdm
import numpy as np
from numpy import sqrt, sign, pi, exp
from numpy.fft import fft, ifft
from scipy import linalg as lin


def derivative_coeff(rank: int = 1, order: int = None) -> "array":
    """Computes the coefficients of a derivaive estimator

    Args:
        rank
            Rank of the derivative
        order
            Order of the Taylor serie used to estimate the derivative. Shall be >= rank
            The default value is *rank*

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


def get_window(win, n: int) -> np.array:
    from scipy.signal import get_window

    w = get_window(win, n)
    nrm = np.sum(w) / n

    return w / nrm


def phase_unfold(sig: np.array, eps: float = 1e-9) -> np.array:
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


def analyse_DV(
    wavelength: float,
    period: float,
    dist0: float,
    damb: float,
    vrad0: float,
    vamb: float,
    seq: "DSPSignal",
    rxsig: "DSPSignal",
    nv: int,
    n_integration: int = -1,
    coherent: bool = True,
    progress_bar: bool = True,
    corr_window: str = "hamming",
) -> "DSPSpectrogram":
    """Distance / velocity analysis for acquisition

    Args:
      wavelength (m)
        Wavelength of the carrier
      period (s)
        Window length
      dist0 (m)
        Center of the distance research domain
      damb (m)
        Width of the distance research domain
      vrad0 (m/s)
        Center of the velocity research domain
      vamb (m/s)
        Width of the velocity research domain
      n_integration
        Number of period to sum. A value of -1 means to sum everything
      seq
        Local replica of the signal
      rxsig
        Received signal to be analysed
      nv
        Number of velocity hypothesis to be tested
      progress_bar
        To turn on the display of a progress bar
      corr_window
        Window to be used for correlation

    Returns:
      The spectrogram of the analysis

    """
    from ..constants import c
    from .DSPSpectrogram import DSPSpectrogram

    if nv % 2 == 0:
        nv += 1

    # Number of samples shift at each period
    vrad_max = max(np.abs(vrad0 + vamb / 2), np.abs(vrad0 - vamb / 2))
    ns_shift = period * vrad_max / c / rxsig.samplingPeriod
    if ns_shift > 1:
        raise AssertionError("Too much sample shift: %.3f" % ns_shift)

    # Number of sample in a period
    nsig = int(period / rxsig.samplingPeriod + 1)

    # Number of samples in a distance window
    nd = int(damb / c / rxsig.samplingPeriod + 1)

    # Sample index of the center distance window
    n0 = int(dist0 / c / rxsig.samplingPeriod)

    kmin = max(n0 - nd // 2, 0)
    kmax = min(n0 - nd // 2 + nd, nsig - 1)
    img = np.empty((kmax - kmin, nv), dtype=np.complex128)

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
            offset=kmin * rxsig.samplingPeriod,
            coherent=coherent,
            window_duration=(kmax - kmin) * rxsig.samplingPeriod,
        )

        img[:, kv] = zi.y_serie

    spg = DSPSpectrogram(
        name="spg",
        samplingXStart=tab_v[0] - vrad0,
        samplingXPeriod=tab_v[1] - tab_v[0],
        samplingYStart=zi.samplingStart,
        samplingYPeriod=zi.samplingPeriod,
        img=img,
    )
    spg.name_of_x_var = "Radial Velocity (%.1f m/s delta)" % vrad0
    spg.unit_of_x_var = "m/s"
    spg.name_of_y_var = "Delay"
    spg.unit_of_y_var = "s"

    return spg


def shift(register, feedback, output):
    """GPS Shift Register

    :param list feedback: which positions to use as feedback (1 indexed)
    :param list output: which positions are output (1 indexed)
    :returns output of shift register:

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
) -> "DSPSignal":
    """Builds Gold sequence

    Args:
        name
            Name of the signal
        repeat
            Number of copies of a 1023 Gold sequence
        sv
            Identifier of the SV. Can be either the PRN number (int), or the code tap selection (list of 2 int)
        samplingStart (s)
            First date of the sample of the signal
        chip_rate (Hz)
            Sampling frequency of the signal
        sampling_factor
            Factor so that fs = sampling_factor*chip_rate
        bitmap
            List of 2 values to map the bits on. [0, 1] returns a sequence with 0 and 1

    Returns:
        The :class:`blocksim.dsp.DSPSignal`. All the samples are +1 or -1

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
        samplingStart=0,
        samplingPeriod=1 / chip_rate / sampling_factor,
        y_serie=seq,
        dtype=np.int64,
    )
    return sig


def zadoff_chu(u, n):
    k = np.arange(n)
    return exp(-1j * pi * u * k * (k + 1) / n)


def createZadoffChu(
    name: str,
    n_zc: int,
    u: int,
    sampling_freq: float,
    samplingStart: float = 0,
) -> "DSPSignal":
    """Builds Zadoff-Chu sequence

    Args:
        name
        Name of the signal
        n_zc
        Length of the Zadoff-Chu sequence
        u
        Index of the Zadoff-Chu sequence

    Returns:
        The :class:`blocksim.dsp.DSPSignal`

    """
    from .DSPSignal import DSPSignal

    seq = zadoff_chu(u, n_zc)
    sig = DSPSignal(
        name=name, samplingStart=0, samplingPeriod=1 / sampling_freq, y_serie=seq
    )
    return sig
