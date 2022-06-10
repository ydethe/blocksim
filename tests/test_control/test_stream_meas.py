import sys
from pathlib import Path
import unittest

import numpy as np
from matplotlib import pyplot as plt
import pytest

from blocksim.control.Sensors import StreamCSVSensors
from blocksim.control.SetPoint import Step
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestStreamMeas(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_stream_csv_meas(self):
        pth = Path(__file__).parent / "test_stream_meas.csv"
        cpt = StreamCSVSensors("cpt", pth=str(pth))
        cpt.loadFile()

        stp = Step(name="stp", cons=np.array([1.0]), snames=["x"])

        sim = Simulation()
        sim.addComputer(cpt)
        sim.addComputer(stp)

        tps = np.arange(0, 2, 0.05)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        fig = self.plotVerif(
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
        return fig.render()


if __name__ == "__main__":
    unittest.main()
    exit(0)

    from blocksim.graphics import showFigures

    a = TestStreamMeas()
    a.test_stream_csv_meas()

    showFigures()
