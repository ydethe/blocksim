import sys
import os
import unittest

import numpy as np
from numpy import cos, sin, sqrt, exp
from scipy.integrate import ode
from matplotlib import pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.Node import Frame, Input, Output, AComputer, connect


class SetPoint(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineOutput("setpoint", initial_state=np.array([1]))

    def updateAllOutput(self, frame: Frame):
        pass


class Controller(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("setpoint")
        self.defineInput("estimation")
        self.defineOutput("command", initial_state=np.array([0]))

    def updateAllOutput(self, frame: Frame):
        inp = self.getInputByName("setpoint")
        stp = self.getDataFromInput(frame, inp.getID())

        inp = self.getInputByName("estimation")
        yest, vest = self.getDataFromInput(frame, inp.getID())

        otp = self.getOutputByName("command")

        k = 10
        m = 1

        a = 8
        P = -k + 3 * a ** 2 * m

        u = P * (yest - stp)

        otp.setData(-u)


class System(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("command")
        self.defineOutput("output", initial_state=np.array([0, 0]))

    def updateAllOutput(self, frame: Frame):
        inp = self.getInputByName("command")
        (u,) = self.getDataFromInput(frame, inp.getID())

        otp = self.getOutputByName("output")
        t0 = frame.getStartTimeStamp()
        t1 = frame.getStopTimeStamp()
        y0 = otp.getDataForFrame(frame)

        k = 10
        f = 5
        m = 1

        def fct(t, y, u):
            yp, vp = y
            a = (-f * vp - k * yp + u) / m
            dy = np.array([vp, a])
            return dy

        r = ode(fct).set_integrator("zvode", method="bdf")
        r.set_initial_value(y0, t0).set_f_params(u).set_jac_params(u)
        y1 = r.integrate(t1)

        otp.setData(y1)


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
        stp = SetPoint("stp")
        ctl = Controller("ctl")
        sys = System("sys")

        connect(
            computer_src=ctl,
            output_name="command",
            computer_dst=sys,
            intput_name="command",
        )
        connect(
            computer_src=sys,
            output_name="output",
            computer_dst=ctl,
            intput_name="estimation",
        )
        connect(
            computer_src=stp,
            output_name="setpoint",
            computer_dst=ctl,
            intput_name="setpoint",
        )

        frame = Frame()
        stp.reset(frame)
        ctl.reset(frame)
        sys.reset(frame)

        frame0 = frame.copy()
        self.assertEqual(frame, frame0)

        tps = np.arange(0, 2, 0.001)
        ns = len(tps)
        x = np.empty(ns)
        x[0], _ = sys.getDataForOuput(frame, name="output")
        for k in range(1, ns):
            dt = tps[k] - tps[k - 1]
            frame.updateByStep(dt)
            self.assertNotEqual(frame, frame0)
            x[k], v = np.real(sys.getDataForOuput(frame, name="output"))

        x_ref = plotAnalyticsolution(tps, cons=1)
        self.assertAlmostEqual(np.max(np.abs(x - x_ref)), 0, delta=2e-4)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.plot(tps, x, label="bs")
        axe.plot(tps, x_ref, label="analytic")
        axe.plot(tps, x * 0 + 1, label="setpoint")
        axe.grid(True)
        axe.legend(loc="best")

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestSimpleControl()
    a.test_simple_control()

    plt.show()
