import os
import sys
from pathlib import Path
from unittest.mock import patch
from pathlib import Path

from blocksim.tools.gnss_sim import sim, mtcl, plot

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestGNSSSimMain(TestBase):
    @patch("matplotlib.pyplot.show")
    def test_cmd_mtcl(self, mock_pyplot):
        rt = Path(__file__).parent
        mtcl(rt / "config" / "cfg_mtcl_ranging.yml")

    @patch("matplotlib.pyplot.show")
    def test_cmd_sim(self, mock_pyplot):
        rt = Path(__file__).parent
        sim(rt / "config/cfg_simu_ranging.yml")

    @patch("matplotlib.pyplot.show")
    def test_cmd_plot(self, mock_pyplot):
        rt = Path(__file__).parent
        plot(rt / "config/cfg_long_galileo.yml", dop=True, save="", simid=-1)
        plot(rt / "config/cfg_long_galileo.yml", polar=True, save="", simid=-1)
        plot(rt / "config/cfg_long_galileo.yml", mtcl=True, save="", simid=-1)


if __name__ == "__main__":
    a = TestGNSSSim()
    a.test_cmd_mtcl()
    # a.test_cmd_plot()
