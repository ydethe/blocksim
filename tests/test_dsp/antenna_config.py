# antenna_config.py
import os
from pathlib import Path
from pickle import load

from numpy import pi, cos

from blocksim.constants import c

root = Path(__file__).parent

name = "ant"
freq = c / 0.3
altitude = 10000.0
hpbw = pi / 2
coeff_pth = root / "coeff.pkl"

if os.path.exists(coeff_pth):
    with open(coeff_pth, "rb") as f:
        coefficients = load(f)


def th_profile(th):
    return altitude / cos(th)


def mapping(k):
    s = c / freq * 0.5
    N = 3
    q = k % N
    p = (k - q) // N
    return p * s, q * s
