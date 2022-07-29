import sys
from pathlib import Path
import unittest

import pytest
import numpy as np
from numpy import sqrt

from blocksim import logger
from blocksim.dsp.OFDMA import OFDMMapping, OFDMDemapping
from blocksim.graphics.BFigure import FigureFactory

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestOFDM(TestBase):
    def setUp(self):
        K = 12
        listCarriers = np.arange(K)
        self.allCarriers = K
        self.pilotCarriers = listCarriers[::3]  # Pilots is every (K/P)th carrier.
        self.dataCarriers = np.delete(listCarriers, self.pilotCarriers)
        self.pilotValue = 3 + 3j
        np.random.seed(155462157)

        nsymb = 3

        # Random QPSK payload
        inv_sq_2 = 1 / sqrt(2)
        self.data = (
            (
                np.random.randint(low=0, high=2, size=(len(self.dataCarriers), nsymb))
                * 2
                - 1
            )
            * inv_sq_2
            * 1j
        )
        self.data += (
            np.random.randint(low=0, high=2, size=(len(self.dataCarriers), nsymb)) * 2
            - 1
        ) * inv_sq_2

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 150})
    def test_ofdm_mapping(self):
        ofdm_co = OFDMMapping(
            name="map",
            output_size=2048,
            allCarriers=self.allCarriers,
            pilotCarriers=self.pilotCarriers,
            dataCarriers=self.dataCarriers,
            pilotValue=self.pilotValue,
        )
        ofdm_dec = OFDMDemapping(
            name="demap",
            input_size=2048,
            allCarriers=self.allCarriers,
            pilotCarriers=self.pilotCarriers,
            dataCarriers=self.dataCarriers,
            pilotValue=self.pilotValue,
        )

        ofdm_payload = ofdm_co.process(self.data)
        sig = ofdm_dec.flatten(ofdm_payload)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(np.real(sig))

        return fig.render()

    def test_ofdm_demapping(self):
        ofdm_co = OFDMMapping(
            name="map",
            output_size=2048,
            allCarriers=self.allCarriers,
            pilotCarriers=self.pilotCarriers,
            dataCarriers=self.dataCarriers,
            pilotValue=self.pilotValue,
        )
        ofdm_dec = OFDMDemapping(
            name="demap",
            input_size=2048,
            allCarriers=self.allCarriers,
            pilotCarriers=self.pilotCarriers,
            dataCarriers=self.dataCarriers,
            pilotValue=self.pilotValue,
        )

        ofdm_payload = ofdm_co.process(self.data)
        data2 = ofdm_dec.process(ofdm_payload)

        self.assertAlmostEqual(np.max(np.abs(self.data - data2)), 0, delta=1e-9)


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestOFDM()
    a.setUp()
    a.test_ofdm_mapping()
    # a.test_ofdm_demapping()

    showFigures()
