from matplotlib import pyplot as plt
import pytest

from blocksim.tools.config import load_config
from blocksim.tools.computation import mtcl
from blocksim.tools.plot import plot_mtcl, plot_polar_view


from blocksim.testing import TestBase


class TestGNSSSimMTCL(TestBase):
    @classmethod
    def setUpClass(cls):
        cfgd = load_config("tests/test_gnss/config/cfg_mtcl_doppler.yml")
        mtcl(cfgd)

        cfgr = load_config("tests/test_gnss/config/cfg_mtcl_ranging.yml")
        mtcl(cfgr)

    @pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 150})
    def test_mtcl(self):
        cfg = load_config("tests/test_gnss/config/cfg_mtcl_ranging.yml")
        fig = plot_mtcl(cfg)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 150})
    def test_mtcl_vr(self):
        cfg = load_config("tests/test_gnss/config/cfg_mtcl_doppler.yml")
        fig = plot_mtcl(cfg)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=6, savefig_kwargs={"dpi": 150})
    def test_polar_view(self):
        cfg = load_config("tests/test_gnss/config/cfg_mtcl_doppler.yml")

        fig, data = plot_polar_view(cfg)

        return fig.render()


if __name__ == "__main__":
    TestGNSSSimMTCL.setUpClass()
    a = TestGNSSSimMTCL()
    # a.test_mtcl()
    # a.test_mtcl_vr()
    a.test_polar_view()
    plt.show()
