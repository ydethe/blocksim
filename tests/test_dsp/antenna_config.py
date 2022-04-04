# antenna_config.py

from pathlib import Path

from numpy import pi, cos, arcsin

from blocksim_sigspace.constants import c

root = Path(__file__).parent

name = "ant"
freq = c / 0.3
altitude = 10000.0
hpbw = arcsin(1 / 3)
coefficients = root / "coeff.pkl"


def th_profile(th):
    return altitude / cos(th)


def mapping(k):
    s = c / freq * 0.5
    N = 3
    q = k % N
    p = (k - q) // N
    return p * s, q * s
