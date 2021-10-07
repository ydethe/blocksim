import sys
from pathlib import Path
import unittest

import numpy as np
from numpy import cos, sin, sqrt, exp, pi
from matplotlib import pyplot as plt
import pytest

from blocksim.core.Node import Frame
from blocksim.control.SetPoint import (
    Step,
    InterpolatedSetPoint,
    Ramp,
    Rectangular,
    Sinusoid,
)
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestSetPoint(TestBase):
    def test_step(self):
        stp = Step(name="stp", snames=["s0", "s1"], cons=np.array([1, -2]))
        uid = stp.getID()

        sim = Simulation()
        sim.addComputer(stp)
        ret = sim.getComputerById(uid)
        self.assertEqual(ret.getName(), stp.getName())

        tps = np.arange(10)
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        err0 = log.getValue("stp_setpoint_s0") - 1
        err1 = log.getValue("stp_setpoint_s1") + 2

        self.assertAlmostEqual(np.max(np.abs(err0)), 0, delta=1e-10)
        self.assertAlmostEqual(np.max(np.abs(err1)), 0, delta=1e-10)

    def test_interp(self):
        stp = InterpolatedSetPoint(name="stp", snames=["s0", "s1"])
        stp.setInterpolatorForOutput((0,), [0, 1, 2, 3], [1, 2, 4, 8], kind="linear")
        stp.setInterpolatorForOutput((1,), [0, 1, 2, 3], [1, 2, 1, 2], kind="linear")

        sim = Simulation()
        sim.addComputer(stp)
        tps = np.arange(3) + 0.5
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        s0 = log.getValue("stp_setpoint_s0")
        s1 = log.getValue("stp_setpoint_s1")

        s0_ref = np.array([1.5, 3, 6])
        s1_ref = np.array([1.5, 1.5, 1.5])

        self.assertAlmostEqual(np.max(np.abs(s0 - s0_ref)), 0, delta=1e-10)
        self.assertAlmostEqual(np.max(np.abs(s1 - s1_ref)), 0, delta=1e-10)

    def test_ramp(self):
        stp = Ramp(name="stp", snames=["s0", "s1"], slopes=np.array([2, -1j]))

        sim = Simulation()
        sim.addComputer(stp)
        tps = np.arange(0, 10)
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        s0 = log.getValue("stp_setpoint_s0")
        s1 = log.getValue("stp_setpoint_s1")

        s0_ref = 2 * tps
        s1_ref = -1j * tps

        self.assertAlmostEqual(np.max(np.abs(s0 - s0_ref)), 0, delta=1e-10)
        self.assertAlmostEqual(np.max(np.abs(s1 - s1_ref)), 0, delta=1e-10)

    def test_sinusoid(self):
        ssd = Sinusoid(name="ssd", snames=["s0", "s1"])
        ssd.freq = np.array([10.0, 19.0])
        ssd.pha = np.array([0.0, pi / 2])
        ssd.amp = np.array([2.0, 1.0])

        sim = Simulation()
        sim.addComputer(ssd)
        fs = 1.0
        tps = np.arange(0, 10) / fs
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        s0 = log.getValue("ssd_setpoint_s0")
        s1 = log.getValue("ssd_setpoint_s1")

        s0_ref = 2 * sin(2 * pi * 10 * tps)
        s1_ref = sin(2 * pi * 19 * tps + pi / 2)

        self.assertAlmostEqual(np.max(np.abs(s0 - s0_ref)), 0, delta=1e-10)
        self.assertAlmostEqual(np.max(np.abs(s1 - s1_ref)), 0, delta=1e-10)

    def test_door(self):
        stp = Rectangular(name="stp", snames=["s0", "s1"])
        stp.doors = {(0,): (1, 1, 0, 2), (1,): (2, 2, 1, 3)}

        sim = Simulation()
        sim.addComputer(stp)
        tps = np.arange(0, 4, 0.2)
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        s0 = log.getValue("stp_setpoint_s0")
        s1 = log.getValue("stp_setpoint_s1")

        ns = len(tps)
        s0_ref = np.zeros(ns)
        ion = np.where(tps >= 1)[0]
        ion = np.intersect1d(ion, np.where(tps < 2)[0])
        s0_ref[ion] = 1

        s1_ref = np.ones(ns)
        ion = np.where(tps >= 2)[0]
        ion = np.intersect1d(ion, np.where(tps < 3)[0])
        s1_ref[ion] = 2

        self.assertAlmostEqual(np.max(np.abs(s0 - s0_ref)), 0, delta=1e-10)
        self.assertAlmostEqual(np.max(np.abs(s1 - s1_ref)), 0, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
