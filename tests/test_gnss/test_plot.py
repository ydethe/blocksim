from pathlib import Path

from munch import munchify
import numpy as np
from numpy import pi
import pytest

from blocksim.graphics import showFigures
from blocksim.graphics.GraphicSpec import AxeProjection
from blocksim.graphics.BFigure import FigureFactory
from blocksim.loggers.Logger import Logger
from blocksim.tools.config import load_config
from blocksim.tools.plot import plot_polar_view, plot_histogram


from blocksim.testing import TestBase


class TestGNSSSimPlot(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=8, savefig_kwargs={"dpi": 150})
    def test_polar(self, fic_cfg="tests/test_gnss/config/cfg_polar_gnss.yml"):
        # http://www.taroz.net/GNSS-Radar.html#
        cfg = load_config(fic_cfg)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 2)
        axe = fig.add_baxe(title=cfg.cfg_file, spec=gs[0, 0], projection=AxeProjection.NORTH_POLAR)
        plot_polar_view(cfg, axe=axe)

        axe = fig.add_baxe(title="", spec=gs[0, 1])
        axe.plot(Path("tests/baseline/position_radar.png"))

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_elev_hist(self):
        log = Logger()
        # Simulate 2 sat at 5 and 40 deg elevation
        tps = np.arange(100)
        for t in tps:
            log.log(name="tkr_obscoord_elev0", val=5 * pi / 180, unit="rad")
            log.log(name="tkr_obscoord_elev1", val=40 * pi / 180, unit="rad")
            log.log(name="t", val=t, unit="s")
        log.export("/tmp/test_log.pkl")

        cfg = munchify({"setup": {"logfile": "/tmp/test_log.pkl"}})
        fig = plot_histogram(cfg=cfg, bin=5)

        return fig.render()


if __name__ == "__main__":
    a = TestGNSSSimPlot()
    a.test_polar()
    # a.test_elev_hist()
    showFigures()
