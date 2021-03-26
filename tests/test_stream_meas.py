import os
import sys
import unittest

import numpy as np
import pytest

from blocksim.control.Sensors import StreamCSVSensors
from blocksim.control.SetPoint import Step
from blocksim.Simulation import Simulation


sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestStreamMeas(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_stream_csv_meas(self):
        cpt = StreamCSVSensors("cpt", pth="tests/test_stream_meas.csv")
        cpt.loadFile()

        stp = Step(name="stp", cons=np.array([1.0]), snames=["x"])

        sim = Simulation()
        sim.addComputer(cpt)
        sim.addComputer(stp)

        tps = np.arange(0, 2, 0.05)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        return self.plotVerif(
            "Figure 1",
            [
                {"var": "stp_setpoint_x", "label": "set point"},
                {
                    "var": "cpt_measurement_x",
                    "label": "measure",
                    "linestyle": "",
                    "marker": "+",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
