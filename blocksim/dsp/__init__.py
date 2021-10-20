import numpy as np
from numpy import sqrt, sign, pi, exp
from numpy.fft import fft, ifft


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
