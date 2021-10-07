import sys
from pathlib import Path
import unittest

import pytest
import numpy as np
from numpy import exp, pi
import sk_dsp_comm.digitalcom as dc
from matplotlib import pyplot as plt

from blocksim import logger
from blocksim.dsp.utils import createParallelBitstream
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.FEC import FECCoder, FECDecoder
from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping
from blocksim.dsp.OFDMA import OFDMMapping, OFDMDemapping
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestChaine(TestBase):
    def run_nbiot_sim(self):
        fs = 1 / (1e-3 / 14)
        ntot = 101

        sim = Simulation()

        tps = createParallelBitstream(
            sim=sim, number=6, samplingPeriod=1 / fs, size=ntot
        )

        fec = FECCoder(name="fec", output_size=18)
        sim.addComputer(fec)

        mapping = [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]
        qpsk = PSKMapping(name="qpsk", mapping=mapping, output_size=9)
        sim.addComputer(qpsk)

        K = 12
        listCarriers = np.arange(K)
        allCarriers = K
        pilotCarriers = [1, 5, 9]  # Pilots is every (K/P)th carrier.
        dataCarriers = np.delete(listCarriers, pilotCarriers)
        pilotValue = 3 + 3j
        ofdm_co = OFDMMapping(
            name="ofmd",
            output_size=2048,
            allCarriers=allCarriers,
            pilotCarriers=pilotCarriers,
            dataCarriers=dataCarriers,
            pilotValue=pilotValue,
        )
        sim.addComputer(ofdm_co)

        ofdm_dec = OFDMDemapping(
            name="demap",
            input_size=2048,
            allCarriers=allCarriers,
            pilotCarriers=pilotCarriers,
            dataCarriers=dataCarriers,
            pilotValue=pilotValue,
        )
        sim.addComputer(ofdm_dec)

        qpsk_dec = PSKDemapping(name="deqpsk", mapping=mapping, output_size=18)
        sim.addComputer(qpsk_dec)

        fec_dec = FECDecoder(name="defec", output_size=6)
        sim.addComputer(fec_dec)

        sim.connect("grp.grouped", "fec.raw")
        sim.connect("fec.coded", "qpsk.input")
        sim.connect("qpsk.output", "ofmd.input")
        sim.connect("ofmd.output", "demap.input")
        sim.connect("demap.output", "deqpsk.input")
        sim.connect("deqpsk.output", "defec.coded")

        sim.simulate(tps, progress_bar=True)


if __name__ == "__main__":
    # unittest.main()

    a = TestChaine()
    a.run_nbiot_sim()
