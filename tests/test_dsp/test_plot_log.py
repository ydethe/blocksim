import sys
from pathlib import Path
import unittest

import pytest
import numpy as np

from blocksim import logger

from blocksim_sigspace.graphics import plotBER

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestPlotLog(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 200})
    def test_plot_log(self):
        pth = Path(__file__).parent / "ber_snr.txt"
        fig = plotBER(str(pth), output="")

        return fig


if __name__ == "__main__":
    unittest.main()
