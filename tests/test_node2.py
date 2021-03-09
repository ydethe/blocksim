import sys
import os
import unittest

import numpy as np
from numpy import cos, sin, sqrt, exp
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.Node import Frame, Input, Output, AComputer, connect


class SetPoint(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineOutput("setpoint", initial_state=np.array([0]))

    def updateAllOutput(self, frame: Frame):
        pass


class Controller(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("setpoint")
        self.defineInput("estimation")
        self.defineOutput("command", initial_state=np.array([0]))
        self.defineOutput("state", initial_state=np.array([0]))

    def updateAllOutput(self, frame: Frame):
        inp = self.getInputByName("setpoint")
        stp = self.getDataFromInput(frame, inp.getID())

        inp = self.getInputByName("estimation")
        yest, vest = self.getDataFromInput(frame, inp.getID())

        otp = self.getOutputByName("command")
        ste = self.getOutputByName("state")

        (itg,) = ste.getDataForFrame(frame)

        k = 1
        m = 40
        a = 8
        P = -k + 3 * a ** 2 * m
        I = a ** 3 * m
        D = 3 * a * m

        u = P * (yest - stp) + I * itg + D * vest

        dt = frame.getTimeStep()
        itg += (yest - stp) * dt

        ste.setData(np.array([itg]))
        otp.setData(-u)


class System(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("command")
        self.defineOutput("output", initial_state=np.array([-1, 0]))

    def updateAllOutput(self, frame: Frame):
        inp = self.getInputByName("command")
        (u,) = self.getDataFromInput(frame, inp.getID())

        otp = self.getOutputByName("output")
        dt = frame.getTimeStep()
        yp, vp = otp.getDataForFrame(frame)

        k = 1
        m = 40
        y = (
            (cos((sqrt(k) * dt) / sqrt(m)) * (k * yp - u)) / k
            + (sqrt(m) * sin((sqrt(k) * dt) / sqrt(m)) * vp) / sqrt(k)
            + u / k
        )
        v = cos((sqrt(k) * dt) / sqrt(m)) * vp - (
            sin((sqrt(k) * dt) / sqrt(m)) * (k * yp - u)
        ) / (sqrt(k) * sqrt(m))

        otp.setData(np.array([y, v]))


def plotAnalyticsolution(tps):
    P, I, D = 7679, 20480, 960
    k = 1
    m = 40

    p = Polynomial([I, k + P, D, m])
    l1, l2, l3 = p.roots()
    det = (
        l2 * l3 ** 2
        - l1 * l3 ** 2
        - l2 ** 2 * l3
        + l1 ** 2 * l3
        + l1 * l2 ** 2
        - l1 ** 2 * l2
    )
    A1 = (l2 ** 2 - l3 ** 2) / det
    A2 = (l3 ** 2 - l1 ** 2) / det
    A3 = (l1 ** 2 - l2 ** 2) / det

    ns = len(tps)
    x = np.empty(ns)
    for ii in range(ns):
        t = tps[ii]
        dW = -(exp(l1 * t) * l1 * A1 + exp(l2 * t) * l2 * A2 + exp(l3 * t) * l3 * A3)
        x[ii] = np.real(dW)

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

        otp = sys.getOutputByName("output")
        oid1 = otp.getID()

        tps = np.arange(0, 2, 0.05)
        ns = len(tps)
        x = np.empty(ns)
        x[0], _ = sys.getDataForOuput(frame, oid1)
        for k in range(1, ns):
            dt = tps[k] - tps[k - 1]
            frame.updateByStep(dt)
            self.assertNotEqual(frame, frame0)
            x[k], v = sys.getDataForOuput(frame, oid1)

        x_ref = plotAnalyticsolution(tps)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.plot(tps, x, label="bs")
        axe.plot(tps, x_ref, label="analytic")
        axe.plot(tps, x * 0, label="setpoint")
        axe.grid(True)
        axe.legend(loc="best")

        return fig


if __name__ == "__main__":
    unittest.main()

    # a = TestSimpleControl()
    # a.test_simple_control()

    # plt.show()
