import os
import sys
import unittest

import pytest
import numpy as np

from blocksim import logger
from blocksim.plot_log import plot_log

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestPlotLog(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 200})
    def test_plot_log(self):
        fig = plot_log("tests/ber_snr.txt", output="")

        return fig


if __name__ == "__main__":
    unittest.main()
