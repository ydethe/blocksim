"""Group delay of analog and digital filters.

From https://github.com/spatialaudio/group-delay-of-filters

"""
import numpy as np
from scipy.signal import tf2sos, group_delay
from scipy.signal.filter_design import _validate_sos


def group_delayz(b: "array", a: "array", w: "array", fs: float) -> float:
    """Compute the group delya of digital filter.

    Args:
        b: Numerator of a linear filter.
        a: Denominator of a linear filter.
        w: Frequencies (Hz)
        fs: The sampling frequency of the digital system (Hz)

    Returns:
        The group delay (s)

    """
    b, a = map(np.atleast_1d, (b, a))
    if len(a) == 1:
        # scipy.signal.group_delay returns gd in samples thus scaled by 1/fs
        gd = group_delay((b, a), w=w, fs=fs)[1] / fs
    else:
        sos = tf2sos(b, a)
        gd = sos_group_delayz(sos, w, fs)[1]
    return gd


def sos_group_delayz(sos: "array", w: "array", fs: float) -> float:
    """Compute group delay of digital filter in SOS format.

    Args:
        sos: Array of second-order filter coefficients, must have shape
            ``(n_sections, 6)``. Each row corresponds to a second-order
            section, with the first three columns providing the numerator
            coefficients and the last three providing the denominator
            coefficients.
        w: Frequencies (Hz)
        fs: The sampling frequency of the digital system (Hz)

    Returns:
        The group delay (s)

    """
    sos, n_sections = _validate_sos(sos)
    if n_sections == 0:
        raise ValueError("Cannot compute group delay with no sections")
    gd = 0
    for biquad in sos:
        gd += quadfilt_group_delayz(biquad[:3], w, fs)[1]
        gd -= quadfilt_group_delayz(biquad[3:], w, fs)[1]
    return gd


def quadfilt_group_delayz(b: "array", w: "array", fs: float):
    """Compute group delay of 2nd-order digital filter.

    Args:
        b: Coefficients of a 2nd-order digital filter
        w: Frequencies (Hz)
        fs: The sampling frequency of the digital system (Hz)

    Returns:
        The group delay (s)

    """
    W = 2 * np.pi * w / fs
    c1 = np.cos(W)
    c2 = np.cos(2 * W)
    u0, u1, u2 = b**2  # b[0]**2, b[1]**2, b[2]**2
    v0, v1, v2 = b * np.roll(b, -1)  # b[0]*b[1], b[1]*b[2], b[2]*b[0]
    num = (u1 + 2 * u2) + (v0 + 3 * v1) * c1 + 2 * v2 * c2
    den = (u0 + u1 + u2) + 2 * (v0 + v1) * c1 + 2 * v2 * c2
    return 1 / fs * num / den


def zpk_group_delay(z: "array", p: "array", k: float, w: "array", fs: float) -> float:
    """Compute group delay of digital filter in zpk format.

    Args:
        z: Zeroes of a linear filter
        p: Poles of a linear filter
        k: Gain of a linear filter
        w: Frequencies (Hz)
        fs: The sampling frequency of the digital system (Hz)

    Returns:
        The group delay (s)

    """
    gd = 0
    for z_i in z:
        gd += zorp_group_delayz(z_i, w)[1]
    for p_i in p:
        gd -= zorp_group_delayz(p_i, w)[1]
    return gd


def zorp_group_delayz(zorp: np.complex128, w: "array", fs: float) -> float:
    """Compute group delay of digital filter with a single zero/pole.

    Args:
        zorp; Zero or pole of a 1st-order linear filter
        w: Frequencies (Hz)
        fs: The sampling frequency of the digital system (Hz)

    Returns:
        The group delay (s)

    """
    W = 2 * np.pi * w / fs
    r, phi = np.abs(zorp), np.angle(zorp)
    r2 = r**2
    cos = np.cos(W - phi)
    return (r2 - r * cos) / (r2 + 1 - 2 * r * cos)
