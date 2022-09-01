import sys
from pathlib import Path

import numpy as np
from scipy import linalg as lin
import pytest

from blocksim.tools.config import load_config
from blocksim.tools.plot import plot_dop
from blocksim.tools.computation import simu

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestGNSSSimCalcul(TestBase):
    def test_calcul(self, fic_cfg="tests/test_gnss/config/cfg_simu_ranging.yml"):
        cfg = load_config(fic_cfg)
        log = simu(cfg)

        pos = np.array(
            [
                log.getRawValue("UE_estpos_px")[-1],
                log.getRawValue("UE_estpos_py")[-1],
                log.getRawValue("UE_estpos_pz")[-1],
            ]
        )

        # Get reference position
        rpos = np.array(
            [
                log.getRawValue("UE_realpos_px")[-1],
                log.getRawValue("UE_realpos_py")[-1],
                log.getRawValue("UE_realpos_pz")[-1],
            ]
        )
        dp = log.getRawValue("UE_estclkerror_dp")[-1]

        # Check errors
        err = lin.norm(rpos - pos)

        dp_ref = cfg.tracker.dp
        err_dp = np.abs(dp - dp_ref)

        # Check PDOP
        sx, sy, sz = (
            log.getRawValue("UE_estdop_sx")[-1],
            log.getRawValue("UE_estdop_sy")[-1],
            log.getRawValue("UE_estdop_sz")[-1],
        )
        pdop = np.array([sx, sy, sz])
        pdopp = lin.norm(np.real(pdop))
        pdopv = lin.norm(np.imag(pdop))

        self.assertAlmostEqual(err, 0, delta=1e-4)
        self.assertAlmostEqual(err_dp, 0, delta=1e-5)
        self.assertAlmostEqual(pdopp, 1.5989593720500173, delta=1e-4)
        self.assertAlmostEqual(pdopv, 0, delta=1e-10)

    def test_calcul_vr(self, fic_cfg="tests/test_gnss/config/cfg_simu_doppler.yml"):
        cfg = load_config(fic_cfg)
        log = simu(cfg)

        pos = np.array(
            [
                log.getRawValue("UE_estpos_px")[-1],
                log.getRawValue("UE_estpos_py")[-1],
                log.getRawValue("UE_estpos_pz")[-1],
            ]
        )

        # Get reference position
        rpos = np.array(
            [
                log.getRawValue("UE_realpos_px")[-1],
                log.getRawValue("UE_realpos_py")[-1],
                log.getRawValue("UE_realpos_pz")[-1],
            ]
        )

        dv = log.getRawValue("UE_estclkerror_dv")[-1]

        # Check errors
        err = lin.norm(rpos - pos)

        dv_ref = cfg.tracker.dv
        err_dv = np.abs(dv - dv_ref)

        # Check PDOP
        sx, sy, sz = (
            log.getRawValue("UE_estdop_sx")[-1],
            log.getRawValue("UE_estdop_sy")[-1],
            log.getRawValue("UE_estdop_sz")[-1],
        )
        pdop = np.array([sx, sy, sz])
        pdopp = lin.norm(np.real(pdop))
        pdopv = lin.norm(np.imag(pdop))

        self.assertAlmostEqual(err, 0, delta=1e-4)
        self.assertAlmostEqual(err_dv, 0, delta=1e-8)
        self.assertAlmostEqual(pdopp, 10881.29623, delta=5e-2)
        self.assertAlmostEqual(pdopv, 0, delta=1e-10)

    @pytest.mark.mpl_image_compare(tolerance=9, savefig_kwargs={"dpi": 150})
    def test_plot_dop(self):
        cfg = load_config("tests/test_gnss/config/cfg_long_galileo.yml")
        fig = plot_dop(cfg)

        return fig.render()


if __name__ == "__main__":
    from blocksim.graphics import showFigures

    a = TestGNSSSimCalcul()
    # a.test_calcul()
    # a.test_calcul_vr()
    a.test_plot_dop()

    showFigures()
