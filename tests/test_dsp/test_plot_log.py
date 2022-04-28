import sys
from pathlib import Path
import unittest

import pytest
import numpy as np

from blocksim import logger

from blocksim.graphics import plotBER

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestPlotLog(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 150})
    def test_plot_log(self):
        pth = Path(__file__).parent / "ber_snr.txt"
        axe = plotBER(str(pth))

        return axe.figure


if __name__ == "__main__":
    unittest.main()
