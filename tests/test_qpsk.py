import os
import sys
import unittest

import pytest
import numpy as np
from numpy import pi
from numpy import log10, sqrt
from matplotlib import pyplot as plt

from blocksim import logger
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping
from blocksim.Simulation import Simulation

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestQPSK(TestBase):
    def test_qpsk(self):
        mapping = [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]

        qpsk_co = PSKMapping(name="map", mapping=mapping)
        qpsk_dec = PSKDemapping(name="demap", mapping=mapping)

        ntot = 256
        data = np.random.randint(low=0, high=2, size=ntot)

        qpsk_payload = qpsk_co.process(data)
        data2 = qpsk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data[:-2] - data2[2:ntot])), 0, delta=1e-9)

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_qpsk_noise(self):
        mapping = [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]
        ntot = 20 * 1023
        fs = 1.023e6

        sim = Simulation()

        bs = DSPSignal.fromRandom(name="bs", samplingPeriod=1 / fs, size=ntot)
        qpsk_co = PSKMapping(name="map", mapping=mapping)
        # qpsk_dec = QPSKDemapping(name="demap")

        sim.addComputer(bs)
        sim.addComputer(qpsk_co)

        sim.connect("bs.setpoint", "map.input")

        tps = bs.generateXSerie()[:10]
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        print(log.getValue("bs_setpoint_bs"))
        print(log.getValue("map_output_symb"))


if __name__ == "__main__":
    # unittest.main()

    a = TestQPSK()
    # a.test_qpsk()
    a.test_qpsk_noise()

    # plt.show()
