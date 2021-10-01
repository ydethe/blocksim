import os
import sys
import unittest

import pytest
import numpy as np
from numpy import pi, exp, log10, sqrt
from matplotlib import pyplot as plt

from blocksim import logger
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping
from blocksim.dsp.DSPAWGN import DSPAWGN
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

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_qpsk_noise(self):
        mapping = np.array([pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4])
        ntot = 1023
        fs = 1.023e6

        sim = Simulation()

        bs = DSPSignal.fromRandom(name="bs", samplingPeriod=1 / fs, size=ntot)
        qpsk_co = PSKMapping(name="map", mapping=mapping)
        awgn = DSPAWGN(
            name="awgn",
            mean=np.array([0.0]),
            cov=np.array([[0.05]]),
            dtype=np.complex128,
        )
        qpsk_dec = PSKDemapping(name="demap", mapping=mapping)

        sim.addComputer(bs)
        sim.addComputer(qpsk_co)
        sim.addComputer(awgn)
        sim.addComputer(qpsk_dec)

        sim.connect("bs.setpoint", "map.input")
        sim.connect("map.output", "awgn.noiseless")
        sim.connect("awgn.noisy", "demap.input")

        tps = bs.generateXSerie()
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        m = log.getValue("awgn_noisy_n0")
        const = exp(1j * mapping)

        ref = log.getValue("bs_setpoint_bs")
        est = log.getValue("demap_output_bit")

        self.assertAlmostEqual(np.max(np.abs(ref[:-2] - est[2:ntot])), 0, delta=1e-9)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)
        axe.set_aspect("equal")
        axe.scatter(np.real(m), np.imag(m), color="blue", marker="+")
        axe.scatter(np.real(const), np.imag(const), color="red", marker="o")

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestQPSK()
    a.test_qpsk()
    # a.test_qpsk_noise()

    # plt.show()
