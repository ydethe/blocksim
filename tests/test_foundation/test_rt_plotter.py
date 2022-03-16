import os
import sys
from unittest.mock import patch
from pathlib import Path

import numpy as np
from numpy import pi
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase

from blocksim.graphics.RTPlotter import RTPlotter
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.Simulation import Simulation


class TestRTPlotter(TestBase):
    @patch("matplotlib.pyplot.show")
    def test_rtplotter(self, mock_pyplot):
        fig = plt.figure()
        axe = fig.add_subplot(111)

        dt = 0.5
        f0 = 1 / dt / 5
        tps = np.arange(10) * dt
        z = np.exp(1j * 2 * pi * tps * f0)
        sig = DSPSignal(name="sig", samplingStart=0, samplingPeriod=dt, y_serie=z)

        im = {}
        im["z"] = [0], {}

        rtp = RTPlotter(name="rtp", axe=axe, input_map=im)

        sim = Simulation()
        sim.addComputer(sig)
        sim.addComputer(rtp)

        sim.connect("sig.setpoint", "rtp.z")

        ani = sim.simulate(tps, fig=fig)


if __name__ == "__main__":
    a = TestRTPlotter()
    a.setUp()
    a.test_rtplotter()
