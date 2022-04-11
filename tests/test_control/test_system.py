import sys
from pathlib import Path
import unittest

import numpy as np
from numpy import cos, sin, sqrt, exp, pi
from matplotlib import pyplot as plt
import pytest

from blocksim.core.Node import Frame
from blocksim.control.SetPoint import Step
from blocksim.control.System import LTISystem
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase, plotAnalyticsolution


class TestSystem(TestBase):
    def test_ltisystem(self):
        stp = Step("stp", snames=["c"], cons=np.array([0]))
        sys = LTISystem("sys", shape_command=1, snames_state=["x", "dx"])
        k = 10
        f = 5
        m = 1
        sys.matA = np.array([[0, 1], [-k / m, -f / m]])
        sys.matB = np.array([[0, 1 / m]]).T
        sys.setInitialStateForOutput(np.array([1, 0]), "state")

        sim = Simulation()
        sim.addComputer(stp)
        sim.addComputer(sys)
        sim.connect(src_name="stp.setpoint", dst_name="sys.command")

        tps = np.arange(0, 2, 0.01)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        ns = self.log.getDataSize()
        self.assertEqual(ns, len(tps))

        ms = self.log.getMatrixOutput(name="stp_setpoint")
        err = np.max(np.abs(ms))
        self.assertAlmostEqual(err, 0, delta=1e-10)

        ms = self.log.getValueForComputer(comp=stp, output_name="setpoint")
        err = np.max(np.abs(ms))
        self.assertAlmostEqual(err, 0, delta=1e-10)

        x = self.log.getValue("sys_state_x")
        x_ref = plotAnalyticsolution(tps, xv0=(1, 0), cons=0, PID=(0, 0, 0))
        err = np.max(np.abs(x - x_ref))
        self.assertAlmostEqual(err, 0, delta=1e-10)


if __name__ == "__main__":
    a = TestSystem()
    a.setUp()
    a.test_ltisystem()