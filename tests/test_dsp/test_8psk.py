import sys
from pathlib import Path
import unittest

import pytest
import numpy as np
from numpy import pi, exp, log10, sqrt
from matplotlib import pyplot as plt

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.DSPAWGN import DSPAWGN
from blocksim.control.Route import Group
from blocksim.Simulation import Simulation

from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class Test8PSK(TestBase):
    def test_8psk(self):
        mapping = [
            0,
            pi / 4,
            pi / 2,
            3 * pi / 4,
            pi,
            5 * pi / 4,
            3 * pi / 2,
            7 * pi / 4,
        ]

        psk_co = PSKMapping(name="map", mapping=mapping, output_size=1)
        psk_dec = PSKDemapping(name="demap", mapping=mapping, output_size=3)

        ntot = 256
        data = np.random.randint(low=0, high=2, size=(psk_co.mu, ntot))

        qpsk_payload = psk_co.process(data)
        data2 = psk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

    def test_8psk_noise(self):
        mapping = [
            0,
            pi / 4,
            pi / 2,
            3 * pi / 4,
            pi,
            5 * pi / 4,
            3 * pi / 2,
            7 * pi / 4,
        ]
        ntot = 1023
        fs = 1.023e6

        sim = Simulation()

        bs0 = DSPSignal.fromBinaryRandom(
            name="bs0", samplingPeriod=1 / fs, size=ntot, seed=9948457
        )
        bs1 = DSPSignal.fromBinaryRandom(
            name="bs1", samplingPeriod=1 / fs, size=ntot, seed=167847
        )
        bs2 = DSPSignal.fromBinaryRandom(
            name="bs2", samplingPeriod=1 / fs, size=ntot, seed=7338557
        )

        grp_inp = dict()
        grp_inp["in0"] = (1,)
        grp_inp["in1"] = (1,)
        grp_inp["in2"] = (1,)
        grp = Group(
            name="grp", inputs=grp_inp, snames=["g0", "g1", "g2"], dtype=np.int32
        )

        psk_co = PSKMapping(name="map", mapping=mapping, output_size=1)
        awgn = DSPAWGN(
            name="awgn",
            mean=np.array([0.0]),
            cov=np.array([[0.05]]),
            dtype=np.complex128,
        )
        psk_dec = PSKDemapping(name="demap", mapping=mapping, output_size=3)

        sim.addComputer(bs0)
        sim.addComputer(bs1)
        sim.addComputer(bs2)
        sim.addComputer(grp)
        sim.addComputer(psk_co)
        sim.addComputer(awgn)
        sim.addComputer(psk_dec)

        sim.connect("bs0.setpoint", "grp.in0")
        sim.connect("bs1.setpoint", "grp.in1")
        sim.connect("bs2.setpoint", "grp.in2")
        sim.connect("grp.grouped", "map.input")
        sim.connect("map.output", "awgn.noiseless")
        sim.connect("awgn.noisy", "demap.input")

        tps = bs0.generateXSerie()
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        m = log.getValue("awgn_noisy_n0")

        ref = log.getValue("bs0_setpoint_bs0")
        est = log.getValue("demap_output_s0")

        ber = len(np.where(ref != est)[0]) / ntot

        self.assertLess(ber, 2.5e-2)


if __name__ == "__main__":
    # unittest.main()

    a = Test8PSK()
    a.test_8psk()
    a.test_8psk_noise()
