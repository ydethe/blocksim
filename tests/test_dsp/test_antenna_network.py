from pathlib import Path
import sys
from pickle import dump, load

import pytest
import numpy as np
from matplotlib import pyplot as plt

from blocksim.graphics import plotSpectrogram

from blocksim.utils import load_antenna_config
from blocksim.dsp.AntennaNetwork import AntennaNetwork

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestAntennaNetwork(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 150})
    def test_antenna_diagram(self, fill: bool = True):

        config = Path(__file__).parent / "antenna_config.py"
        ac = load_antenna_config(config)

        ant = AntennaNetwork(ac)

        fig = plt.figure()
        gs = fig.add_gridspec(1, 1)

        diag = ant.antennaDiagram()

        plotSpectrogram(diag, spec=gs[0, 0], levels=50, fill="contour")

        return fig


if __name__ == "__main__":
    a = TestAntennaNetwork()

    a.setUp()
    a.test_antenna_diagram()

    showFigures()()
