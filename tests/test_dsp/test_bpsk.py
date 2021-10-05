import sys
from pathlib import Path
import unittest
from collections import OrderedDict

import pytest
import numpy as np
from numpy import pi, exp, log10, sqrt
from matplotlib import pyplot as plt

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping
from blocksim.dsp.DSPAWGN import DSPAWGN
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestBPSK(TestBase):
    def test_bpsk(self):
        mapping = [0, pi]

        psk_co = PSKMapping(name="map", mapping=mapping, output_size=1)
        psk_dec = PSKDemapping(name="demap", mapping=mapping, output_size=1)

        ntot = 256
        data = np.random.randint(low=0, high=2, size=(psk_co.mu, ntot))

        qpsk_payload = psk_co.process(data)
        data2 = psk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_bpsk_noise(self):
        mapping = [0, pi]
        ntot = 1023
        fs = 1.023e6

        sim = Simulation()

        bs0 = DSPSignal.fromBinaryRandom(
            name="bs0", samplingPeriod=1 / fs, size=ntot, seed=9948457
        )

        psk_co = PSKMapping(name="map", mapping=mapping, output_size=1)
        awgn = DSPAWGN(
            name="awgn",
            mean=np.array([0.0]),
            cov=np.array([[0.05]]),
            dtype=np.complex128,
        )
        psk_dec = PSKDemapping(name="demap", mapping=mapping, output_size=1)

        sim.addComputer(bs0)
        sim.addComputer(psk_co)
        sim.addComputer(awgn)
        sim.addComputer(psk_dec)

        sim.connect("bs0.setpoint", "map.input")
        sim.connect("map.output", "awgn.noiseless")
        sim.connect("awgn.noisy", "demap.input")

        tps = bs0.generateXSerie()
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        m = log.getValue("awgn_noisy_n0")

        ref = log.getValue("bs0_setpoint_bs0")
        est = log.getValue("demap_output_s0")

        ber = len(np.where(ref != est)[0]) / ntot

        self.assertLess(ber, 1e-4)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)
        axe.set_aspect("equal")
        axe.scatter(
            np.real(m), np.imag(m), color="blue", marker="+", label="RX symbols"
        )
        psk_co.plotConstellation(axe)
        axe.legend(loc="best")

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestBPSK()
    a.test_bpsk()
    a.test_bpsk_noise()

    plt.show()
