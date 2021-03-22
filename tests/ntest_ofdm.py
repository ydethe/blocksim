import unittest

import pytest
import numpy as np
from numpy import log10, sqrt
import sk_dsp_comm.digitalcom as dc

from blocksim import logger
from blocksim.blocs.blocksimA import blocksimMapping, blocksimDemapping
from tests.TestBase import TestBase


class Testblocksim(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_ofdm_mapping(self):
        ofdm_co = blocksimMapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )
        ofdm_dec = blocksimDemapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )

        nsymb = 3

        # Random QPSK payload
        inv_sq_2 = 1 / sqrt(2)
        data = (
            (
                np.random.randint(low=0, high=2, size=nsymb * len(self.dataCarriers))
                * 2
                - 1
            )
            * inv_sq_2
            * 1j
        )
        data += (
            np.random.randint(low=0, high=2, size=nsymb * len(self.dataCarriers)) * 2
            - 1
        ) * inv_sq_2

        ofdm_payload = ofdm_co.process(data)

        axe = ofdm_co.plotOutput(df_khz=15)

        return axe.figure

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_ofdm_demapping(self):
        ofdm_co = blocksimMapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )
        ofdm_dec = blocksimDemapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )

        nsymb = 3

        # Random QPSK payload
        inv_sq_2 = 1 / sqrt(2)
        data = (
            (
                np.random.randint(low=0, high=2, size=nsymb * len(self.dataCarriers))
                * 2
                - 1
            )
            * inv_sq_2
            * 1j
        )
        data += (
            np.random.randint(low=0, high=2, size=nsymb * len(self.dataCarriers)) * 2
            - 1
        ) * inv_sq_2

        ofdm_payload = ofdm_co.process(data)
        data2 = ofdm_dec.process(ofdm_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

        axe = ofdm_dec.plotOutput()

        return axe.figure

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_ofdm_carriers(self):
        ofdm_co = blocksimMapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )

        axe = ofdm_co.plotCarriers()

        return axe.figure


if __name__ == "__main__":
    unittest.main()
