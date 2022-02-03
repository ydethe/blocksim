import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import inv
import scipy.linalg as lin
from matplotlib import pyplot as plt
import pytest

from blocksim.core.Frame import Frame
from blocksim.control.System import G6DOFSystem
from blocksim.Simulation import Simulation
from blocksim.control.SetPoint import Step, Rectangular

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class Test6DOFSys(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_trans(self):
        sys = G6DOFSystem("sys")
        g0 = -10
        sys.setInitialStateForOutput(
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -2, 1, 3]), output_name="state"
        )
        stp = Step(
            "stp",
            snames=["fx", "fy", "fz", "tx", "ty", "tz"],
            cons=np.array([0, 0, g0 * sys.m, 0, 0, 0]),
        )

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "sys.command")

        tps = np.linspace(0, 4, 400)
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()

        pz = self.log.getValue("sys_state_pz")
        tf = tps[-1]
        self.assertAlmostEqual(pz[-1], g0 * tf**2 / 2, delta=1e-8)

        return self.plotVerif(
            "Figure 1",
            [{"var": "sys_state_px"}],
            [{"var": "sys_state_py"}],
            [{"var": "sys_state_pz"}],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_att_cplxe(self):
        sys = G6DOFSystem("sys")
        g0 = -10
        sys.setInitialStateForOutput(
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -2, 1, 3]), output_name="state"
        )
        stp = Step(
            "stp",
            snames=["fx", "fy", "fz", "tx", "ty", "tz"],
            cons=np.array([0, 0, g0 * sys.m, 0, 0, 0]),
        )

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "sys.command")

        tps = np.linspace(0, 4, 400)
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()

        return self.plotVerif(
            "Figure 1",
            [{"var": "deg(sys_euler_roll)"}],
            [{"var": "deg(sys_euler_pitch)"}],
            [{"var": "deg(sys_euler_yaw)"}],
        )

    def test_att_roll(self):
        sys = G6DOFSystem("sys")
        g0 = -10
        sys.setInitialStateForOutput(
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2 * np.pi / 4, 0, 0]),
            output_name="state",
        )
        stp = Step(
            "stp",
            snames=["fx", "fy", "fz", "tx", "ty", "tz"],
            cons=np.array([0, 0, g0 * sys.m, 0, 0, 0]),
        )

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "sys.command")

        tps = np.linspace(0, 4, 400)
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()

        r = self.log.getValue("sys_euler_roll")
        self.assertAlmostEqual(r[-1], 0, delta=1e-8)

    def test_att_pitch(self):
        sys = G6DOFSystem("sys")
        g0 = -10
        sys.setInitialStateForOutput(
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2 * np.pi / 4, 0]),
            output_name="state",
        )
        stp = Step(
            "stp",
            snames=["fx", "fy", "fz", "tx", "ty", "tz"],
            cons=np.array([0, 0, g0 * sys.m, 0, 0, 0]),
        )

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "sys.command")

        tps = np.linspace(0, 4, 400)
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()

        p = self.log.getValue("sys_euler_pitch")
        self.assertAlmostEqual(p[-1], 0, delta=1e-8)

    def test_att_yaw(self):
        sys = G6DOFSystem("sys")
        g0 = -10
        sys.setInitialStateForOutput(
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2 * np.pi / 4]),
            output_name="state",
        )
        stp = Step(
            "stp",
            snames=["fx", "fy", "fz", "tx", "ty", "tz"],
            cons=np.array([0, 0, g0 * sys.m, 0, 0, 0]),
        )

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "sys.command")

        tps = np.linspace(0, 4, 400)
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()

        y = self.log.getValue("sys_euler_yaw")
        self.assertAlmostEqual(y[-1], 0, delta=1e-8)

    def test_att_torque(self):
        sys = G6DOFSystem("sys")
        T = 1 / 8000
        stp = Step(
            "stp",
            snames=["fx", "fy", "fz", "tx", "ty", "tz"],
            cons=np.array([0, 0, 0, T, 0, 0]),
        )

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "sys.command")

        tps = np.linspace(0, 4, 400)
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()

        r = self.log.getValue("sys_euler_roll")
        tf = tps[-1]
        self.assertAlmostEqual(r[-1], T * tf**2 / 2 / sys.J[0, 0], delta=1e-8)

        frame = Frame()
        vf = np.array([1, 0, 0])
        vb = sys.vecEarthToBody(frame, vf)
        self.assertAlmostEqual(lin.norm(vb - vf), 0.0, delta=1e-8)


if __name__ == "__main__":
    unittest.main()

    # a=Test6DOFSys()
    # a.test_trans()
    # a.test_att_cplxe()
    # a.test_att_roll()
    # a.test_att_pitch()
    # a.test_att_yaw()
    # a.test_att_torque()

    # plt.show()
