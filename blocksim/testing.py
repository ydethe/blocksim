import os
import unittest

from hypothesis import strategies as st
import numpy as np
from numpy import sqrt, cos, sin, exp, pi

from .graphics import plotVerif, showFigures
from .graphics.BFigure import ABFigure


st_lon = st.sampled_from(np.linspace(-pi, pi, 101))
st_lat = st.sampled_from(np.linspace(-pi / 2, pi / 2, 101))
st_leo_alt = st.floats(200e3, 1200e3, allow_nan=False, allow_subnormal=False)
st_elev = st.sampled_from(np.linspace(0, pi / 2, 101))
st_lonlatalt = st.tuples(st_lon, st_lat, st_leo_alt)
st_latlonalt = st.tuples(st_lat, st_lon, st_leo_alt)


def exact(t, yyp, vvp, u):
    k = 10
    f = 5
    m = 1

    w_0 = sqrt(4 * k * m - f**2) / (2 * m)
    x = (
        (
            (4 * f * m**2 * w_0**2 + f**3) * sin(t * w_0)
            + (8 * m**3 * w_0**3 + 2 * f**2 * m * w_0) * cos(t * w_0)
        )
        * yyp
        + (8 * m**3 * vvp * w_0**2 + 2 * f**2 * m * vvp - 4 * f * m * u) * sin(t * w_0)
        - 8 * m**2 * u * w_0 * cos(t * w_0)
        + 8 * m**2 * exp((f * t) / (2 * m)) * u * w_0
    ) / (
        8 * m**3 * exp((f * t) / (2 * m)) * w_0**3
        + 2 * f**2 * m * exp((f * t) / (2 * m)) * w_0
    )
    v = -(
        exp(-(f * t) / (2 * m))
        * (
            (4 * m**2 * w_0**2 + f**2) * sin(t * w_0) * yyp
            + (2 * f * m * vvp - 4 * m * u) * sin(t * w_0)
            - 4 * m**2 * vvp * w_0 * cos(t * w_0)
        )
    ) / (4 * m**2 * w_0)

    return x, v


def plotAnalyticsolution(tps, xv0, cons, PID):
    ns = len(tps)
    dt = tps[1] - tps[0]

    Kprop, Kinteg, Kderiv = PID
    ix = 0

    x = np.empty(ns)
    yp, vp = xv0
    for k in range(ns):
        x[k] = yp
        u = -Kprop * (yp - cons) - Kinteg * ix - Kderiv * vp
        ix += dt * (yp - cons)
        yp, vp = exact(dt, yp, vp, u)

    return x


class TestBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1883647)

    def plotVerif(self, fig_title, *axes) -> ABFigure:
        fig = plotVerif(self.log, fig_title, *axes)

        if "SHOW_PLOT" in os.environ.keys():
            showFigures()

        return fig
