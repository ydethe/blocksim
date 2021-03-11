import sys
import os
import unittest

import numpy as np
from numpy import cos, sin, sqrt, exp
from matplotlib import pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.core.Node import Frame
from blocksim.blocks.SetPoint import Step
from blocksim.blocks.System import ASystem
from blocksim.blocks.Controller import PController
from blocksim.Simulation import Simulation


# source activate "D:\Users\blaudiy\Documents\Mes Outils Personnels\myenv"
class System(ASystem):
    def __init__(self, name: str):
        ASystem.__init__(self, name, nscal_command=1, nscal_state=2)
        self.setInitialStateForOutput(np.zeros(2), "state")

    def transition(self, t: float, y: np.array, u: np.array) -> np.array:
        k = 10
        f = 5
        m = 1

        yp, vp = y
        a = (-f * vp - k * yp + u[0]) / m
        dy = np.array([vp, a])

        return dy


def exact(t, yyp, vvp, u):
    k = 10
    f = 5
    m = 1

    w_0 = sqrt(4 * k * m - f ** 2) / (2 * m)
    x = (
        (
            (4 * f * m ** 2 * w_0 ** 2 + f ** 3) * sin(t * w_0)
            + (8 * m ** 3 * w_0 ** 3 + 2 * f ** 2 * m * w_0) * cos(t * w_0)
        )
        * yyp
        + (8 * m ** 3 * vvp * w_0 ** 2 + 2 * f ** 2 * m * vvp - 4 * f * m * u)
        * sin(t * w_0)
        - 8 * m ** 2 * u * w_0 * cos(t * w_0)
        + 8 * m ** 2 * exp((f * t) / (2 * m)) * u * w_0
    ) / (
        8 * m ** 3 * exp((f * t) / (2 * m)) * w_0 ** 3
        + 2 * f ** 2 * m * exp((f * t) / (2 * m)) * w_0
    )
    v = -(
        exp(-(f * t) / (2 * m))
        * (
            (4 * m ** 2 * w_0 ** 2 + f ** 2) * sin(t * w_0) * yyp
            + (2 * f * m * vvp - 4 * m * u) * sin(t * w_0)
            - 4 * m ** 2 * vvp * w_0 * cos(t * w_0)
        )
    ) / (4 * m ** 2 * w_0)

    return x, v


def plotAnalyticsolution(tps, cons):
    ns = len(tps)
    dt = tps[1] - tps[0]

    P = 182

    x = np.empty(ns)
    yp, vp = 0, 0
    for k in range(ns):
        x[k] = yp
        u = -P * (yp - cons)
        yp, vp = exact(dt, yp, vp, u)

    return x


class TestSimpleControl(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_simple_control(self):
        k = 10
        m = 1
        a = 8
        P = -k + 3 * a ** 2 * m

        stp = Step("stp", cons=np.array([1]))
        ctl = PController("ctl", nscal_estimation=2, coeff_P=P)
        sys = System("sys")

        sim = Simulation()
        sim.addComputer(stp)
        sim.addComputer(ctl)
        sim.addComputer(sys)

        sim.connect(src_name="ctl.command", dst_name="sys.command")
        sim.connect(src_name="sys.state", dst_name="ctl.estimation")
        sim.connect(src_name="stp.setpoint", dst_name="ctl.setpoint")

        tps = np.arange(0, 2, 0.001)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        x = self.log.getValue("sys_state_0")
        x_ref = plotAnalyticsolution(tps, cons=1)
        err = np.max(np.abs(x - x_ref))
        self.assertAlmostEqual(err, 0, delta=2e-4)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        self.plotVerif("sys_state_0", axe, label="bs")
        axe.plot(tps, x_ref, label="analytic")
        self.plotVerif("stp_setpoint_0", axe, label="setpoint")
        axe.legend(loc="best")

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestSimpleControl()
    a.test_simple_control()

    plt.show()
