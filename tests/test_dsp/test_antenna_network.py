from pathlib import Path
import sys
import unittest

import pytest

from blocksim.graphics.BFigure import FigureFactory
from blocksim.graphics.GraphicSpec import AxeProjection
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

        diag = ant.antennaDiagram()

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0], projection=AxeProjection.POLAR)
        axe.plot(diag, levels=50, fill="contour")

        return fig.render()


if __name__ == "__main__":
    unittest.main()
    exit(0)

    from blocksim.graphics import showFigures

    a = TestAntennaNetwork()

    a.setUp()
    a.test_antenna_diagram()

    showFigures()
