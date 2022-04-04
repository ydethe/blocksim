import sys
from pathlib import Path
import unittest

import numpy as np
from matplotlib import pyplot as plt
import pytest

from blocksim.control.System import LTISystem
from blocksim.control.SetPoint import Step
from blocksim.Simulation import Simulation
from blocksim.control.Sensors import LinearSensors

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestSynchro(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_synchro(self):
        # =========================
        # Definition of the system
        # =========================
        m = 1.0  # Mass
        k = (2 * np.pi) ** 2  # Spring rate
        sys = LTISystem("sys", shape_command=1, snames_state=["x", "v"])
        sys.matA = np.array([[0, 1], [-k / m, 0]])
        sys.matB = np.array([[0, 1 / m]]).T
        sys.setInitialStateForOutput(np.array([-1.0, 0.0]), "state")

        # =========================
        # Definition of the set point
        # =========================
        stp = Step(name="stp", snames=["x_cons"], cons=np.array([0]))

        # =========================
        # Definition of the sensors
        # =========================
        cpt = LinearSensors(
            name="cpt", shape_state=(2,), shape_command=(1,), snames=["x", "v"]
        )
        cpt.setMean(np.zeros(2))
        cpt.setCovariance(np.zeros((2, 2)))
        cpt.matC = np.eye(2)
        cpt.matD = np.zeros((2, 1))

        # =========================
        # Definition of the simulation graph
        # =========================
        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(cpt)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "sys.command")
        sim.connect("stp.setpoint", "cpt.command")
        sim.connect("sys.state", "cpt.state")

        # =========================
        # Simulation of the setup
        # =========================
        tps = np.arange(0, 1 + 0.01, 0.01)
        sim.simulate(tps, progress_bar=False)

        # =========================
        # Plotting
        # =========================
        self.log = sim.getLogger()

        err = self.log.getValue("np.abs(sys_state_x-cpt_measurement_x)")
        self.assertAlmostEqual(np.max(err), 0.0, delta=0.0)

        return self.plotVerif(
            "Figure 1",
            [
                {"var": "sys_state_x", "label": "simulation"},
                {
                    "var": "cpt_measurement_x",
                    "label": "measure",
                    "linestyle": "",
                    "marker": "+",
                },
            ],
        )


if __name__ == "__main__":
    # unittest.main()

    a = TestSynchro()
    a.test_synchro()

    plt.show()
