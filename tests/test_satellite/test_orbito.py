import sys
from pathlib import Path
import unittest

import numpy as np
from numpy import testing

from blocksim.utils import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestOrbito(TestBase):
    def test_teme_itrf(self):
        t = 1000.0
        pv_itrf = np.array(
            [
                -1.48138924e07,
                -2.10421715e07,
                -1.46534295e07,
                2.70050410e03,
                -1.76191617e02,
                -2.47601263e03,
            ]
        )
        pv_teme = itrf_to_teme(t_epoch=t, pv_itrf=pv_itrf)
        tst = teme_to_itrf(t_epoch=t, pv_teme=pv_teme)
        testing.assert_allclose(actual=tst, desired=pv_itrf, rtol=1e-10, equal_nan=True)

        pv_itrf = teme_to_itrf(t_epoch=t, pv_teme=pv_teme)
        tst = itrf_to_teme(t_epoch=t, pv_itrf=pv_itrf)
        testing.assert_allclose(actual=tst, desired=pv_teme, rtol=1e-10, equal_nan=True)

    def test_anomaly(self):
        ecc = 0.02
        mano = 0.2

        E = anomaly_mean_to_ecc(ecc, mano)
        M = anomaly_ecc_to_mean(ecc, E)
        self.assertAlmostEqual(M, mano, delta=1e-10)

        v = anomaly_mean_to_true(ecc, mano)
        M = anomaly_true_to_mean(ecc, v)
        self.assertAlmostEqual(M, mano, delta=1e-10)

        E = anomaly_true_to_ecc(ecc, v)
        v2 = anomaly_ecc_to_true(ecc, E)
        self.assertAlmostEqual(v2, v, delta=1e-10)

    def test_teme_orbital(self):
        a = 7e6
        ecc = 0.02
        argp = 0.4
        inc = 0.3
        mano = 0.2
        node = 0.1

        pv = orbital_to_teme(a, ecc, argp, inc, mano, node)
        orb_param = teme_to_orbital(pv)
        testing.assert_allclose(
            actual=orb_param,
            desired=[a, ecc, argp, inc, mano, node],
            rtol=5e-10,
            equal_nan=True,
        )
        pv2 = orbital_to_teme(a, ecc, argp, inc, mano, node)
        testing.assert_allclose(actual=pv2, desired=pv, rtol=5e-10, equal_nan=True)


if __name__ == "__main__":
    unittest.main()

    # a = TestOrbito()
    # a.test_orbito()
