import sys
from pathlib import Path
import unittest

import numpy as np
from numpy import cos, sin, sqrt, exp
from matplotlib import pyplot as plt
import pytest

from blocksim.core.Node import Frame
from blocksim.control.SetPoint import Step
from blocksim.control.System import ASystem
from blocksim.control.Controller import PIDController
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase, plotAnalyticsolution


ref_repr = """   ========================
   |        'sys'         |
   |        System        |
   ========================
   |                      |
-> command(1,)    state(2,) ->
   |                      |
   ========================
"""


class System(ASystem):
    def __init__(self, name: str):
        ASystem.__init__(self, name, shape_command=1, snames_state=["x", "v"])
        self.setInitialStateForOutput(np.zeros(2), "state")

    def transition(self, t: float, x: np.array, u: np.array) -> np.array:
        k = 10
        f = 5
        m = 1

        yp, vp = x
        a = (-f * vp - k * yp + u[0]) / m
        dx = np.array([vp, a])

        return dx


class TestSimpleControl(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_simple_control(self):
        k = 10
        m = 1
        a = 8
        P = -k + 3 * a**2 * m
        I = a**3 * m
        D = 3 * a * m

        stp = Step("stp", snames=["c"], cons=np.array([1]))
        ctl = PIDController("ctl", shape_estimation=2, snames=["u"], coeffs=(P, I, D))
        sys = System("sys")
        self.assertEqual(str(sys), ref_repr)

        sim = Simulation()
        sim.addComputer(stp)
        sim.addComputer(ctl)
        sim.addComputer(sys)

        sim.connect(src_name="ctl.command", dst_name="sys.command")
        sim.connect(src_name="sys.state", dst_name="ctl.estimation")
        sim.connect(src_name="stp.setpoint", dst_name="ctl.setpoint")

        tps = np.arange(0, 2, 0.01)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        x = self.log.getValue("sys_state_x")
        x_ref = plotAnalyticsolution(tps, xv0=(0, 0), cons=1, PID=(182, 512, 24))
        err = np.max(np.abs(x - x_ref))
        self.assertAlmostEqual(err, 0, delta=1e-10)

        return self.plotVerif(
            "Figure 1",
            [{"var": "sys_state_x"}, {"var": "stp_setpoint_c"}],
        )


if __name__ == "__main__":
    # unittest.main()

    a = TestSimpleControl()
    a.test_simple_control()

    plt.show()
