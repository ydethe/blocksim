import os
import sys
import unittest

import numpy as np
import pytest

from blocksim.blocks.Sensors import StreamCSVSensors
from blocksim.blocks.Controller import LQRegulator
from blocksim.blocks.SetPoint import Step
from blocksim.Simulation import Simulation
from blocksim.blocks.System import LTISystem


sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestStreamMeas(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_stream_csv_meas(self):
        m = 1.0  # Mass
        k = 40.0  # Spring rate
        sys = LTISystem("sys", shape_command=(1,), snames_state=["x", "v"])
        sys.matA = np.array([[0, 1], [-k / m, 0]])
        sys.matB = np.array([[0, 1 / m]]).T
        sys.setInitialStateForOutput(np.array([-1.0, 0.0]), "state")

        cpt = StreamCSVSensors("cpt", pth="tests/test_stream_meas.csv")

        ctl = LQRegulator(
            name="ctl", shape_setpoint=(1,), shape_estimation=(2,), snames=["u"]
        )
        ctl.matA = sys.matA
        ctl.matB = sys.matB
        C = np.hstack((sys.matB, sys.matA @ sys.matB))
        ctl.matQ = C.T @ C * 100
        ctl.matR = np.eye(1)
        ctl.matC = np.array([[1, 0]])
        ctl.matD = np.array([[0]])
        ctl.computeGain()

        stp = Step(name="stp", cons=np.array([1.0]), snames=["x"])

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(ctl)
        sim.addComputer(cpt)
        sim.addComputer(stp)
        sim.connect("stp.setpoint", "ctl.setpoint")
        sim.connect("ctl.command", "sys.command")
        sim.connect("cpt.measurement", "ctl.estimation")

        tps = np.arange(0, 2, 0.05)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        err = self.log.getValue("np.abs(sys_state_x-stp_setpoint_x)")
        self.assertAlmostEqual(err[-1], 0, delta=2e-2)

        return self.plotVerif(
            "Figure 1",
            [
                {"var": "sys_state_x", "label": "simulation"},
                {"var": "stp_setpoint_x", "label": "set point"},
                {"var": "cpt_measurement_x", "label": "measure", "linestyle": "", "marker": "+"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
