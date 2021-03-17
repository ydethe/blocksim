import os
import sys
import unittest
from typing import Iterable

import numpy as np
import scipy.linalg as lin
from matplotlib import pyplot as plt
import pytest

from blocksim.blocks.System import ASystem, G6DOFSystem
from blocksim.blocks.Controller import (
    AController,
    LQRegulator,
    AntiWindupPIDController,
)
from blocksim.Simulation import Simulation
from blocksim.blocks.SetPoint import Step, Rectangular
from blocksim.utils import quat_to_matrix, quat_to_euler
from blocksim.blocks.Route import Group, Split

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase
from Quadcopter.Quadri import Quadri
from Quadcopter.AttPilot import AttPilot
from Quadcopter.Motor import Motor
from TestBase import TestBase


class TestQuad(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_motor(self):
        mot = Motor(0)
        sys = Quadri(mot)
        ctl = AntiWindupPIDController("ctl", shape_estimation=(2,), snames=["u"])
        ctl.D = 0.0
        tau = 50e-3
        ctl.I = mot.km / tau
        ctl.P = ctl.I * mot.Jr * mot.R / mot.km ** 2
        ctl.Ks = 10.0
        ctl.Umin = -mot.Umax
        ctl.Umax = mot.Umax
        s_eq = sys.getEquilibriumSpeed()
        stp = Step("stp", cons=np.ones(1) * s_eq, snames=["c"])

        sim = Simulation()
        sim.addComputer(ctl)
        sim.addComputer(mot)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "ctl.setpoint")
        sim.connect("ctl.command", "mot0.command")
        sim.connect("mot0.vel", "ctl.estimation")

        tps = np.arange(0, 4, 0.01)
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()

        s0 = self.log.getValue("mot0_state_s")
        self.assertAlmostEqual(s0[-1], s_eq, delta=5.5)

        return self.plotVerif(
            "Figure 1",
            [{"var": "stp_setpoint_c"}, {"var": "mot0_state_s"}],
            [{"var": "ctl_command_u"}],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_quad(self):
        mot1 = Motor(1)
        mot2 = Motor(2)
        mot3 = Motor(3)
        mot4 = Motor(4)
        grp = Group(
            "grp",
            inputs={"in1": (1,), "in2": (1,), "in3": (1,), "in4": (1,)},
            snames=["gs1", "gs2", "gs3", "gs4"],
        )
        sys = Quadri(mot1)
        stp = Step("stp", cons=np.ones(1) * 8, snames=["u"])

        sim = Simulation()
        sim.addComputer(mot1)
        sim.addComputer(mot2)
        sim.addComputer(mot3)
        sim.addComputer(mot4)
        sim.addComputer(grp)
        sim.addComputer(stp)
        sim.addComputer(sys)

        sim.connect("stp.setpoint", "mot1.command")
        sim.connect("stp.setpoint", "mot2.command")
        sim.connect("stp.setpoint", "mot3.command")
        sim.connect("stp.setpoint", "mot4.command")
        sim.connect("mot1.state", "grp.in1")
        sim.connect("mot2.state", "grp.in2")
        sim.connect("mot3.state", "grp.in3")
        sim.connect("mot4.state", "grp.in4")
        sim.connect("grp.grouped", "sys.command")

        tps = np.arange(0, np.sqrt(2 * 100 / 9.81), 0.05)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        pz = self.log.getValue("sys_state_pz")
        self.assertAlmostEqual(pz[-1], 16.66, delta=1e-2)

        r = self.log.getValue("sys_euler_roll")
        p = self.log.getValue("sys_euler_pitch")
        y = self.log.getValue("sys_euler_yaw")
        self.assertAlmostEqual(np.max(np.abs(r)), 0, delta=1e-9)
        self.assertAlmostEqual(np.max(np.abs(p)), 0, delta=1e-9)
        self.assertAlmostEqual(np.max(np.abs(y)), 0, delta=1e-9)

        return self.plotVerif(
            "Figure 1",
            [{"var": "sys_state_pz"}],
            [
                {"var": "sys_euler_roll*180/np.pi"},
                {"var": "sys_euler_pitch*180/np.pi"},
                {"var": "sys_euler_yaw*180/np.pi"},
            ],
        )


class TestCmdAtt(TestBase):
    @classmethod
    def setUpClass(cls):
        """get_some_resource() is slow, to avoid calling it for each test use setUpClass()
        and store the result as class variable
        """
        super(TestCmdAtt, cls).setUpClass()

        mot0 = Motor(0)
        mot1 = Motor(1)
        mot2 = Motor(2)
        mot3 = Motor(3)

        tau = 50e-3
        Ks = 0.0

        ctl_mot0 = AntiWindupPIDController(
            "ctlmot0", snames=["u"], shape_estimation=(2,)
        )
        ctl_mot0.D = 0.0
        ctl_mot0.I = mot0.km / tau
        ctl_mot0.P = ctl_mot0.I * mot0.Jr * mot0.R / mot0.km ** 2
        ctl_mot0.Ks = Ks
        ctl_mot0.Umin = -mot0.Umax
        ctl_mot0.Umax = mot0.Umax

        ctl_mot1 = AntiWindupPIDController(
            "ctlmot1", snames=["u"], shape_estimation=(2,)
        )
        ctl_mot1.D = 0.0
        ctl_mot1.I = mot0.km / tau
        ctl_mot1.P = ctl_mot1.I * mot0.Jr * mot0.R / mot0.km ** 2
        ctl_mot1.Ks = Ks
        ctl_mot1.Umin = -mot0.Umax
        ctl_mot1.Umax = mot0.Umax

        ctl_mot2 = AntiWindupPIDController(
            "ctlmot2", snames=["u"], shape_estimation=(2,)
        )
        ctl_mot2.D = 0.0
        ctl_mot2.I = mot0.km / tau
        ctl_mot2.P = ctl_mot2.I * mot0.Jr * mot0.R / mot0.km ** 2
        ctl_mot2.Ks = Ks
        ctl_mot2.Umin = -mot0.Umax
        ctl_mot2.Umax = mot0.Umax

        ctl_mot3 = AntiWindupPIDController(
            "ctlmot3", snames=["u"], shape_estimation=(2,)
        )
        ctl_mot3.D = 0.0
        ctl_mot3.I = mot0.km / tau
        ctl_mot3.P = ctl_mot3.I * mot0.Jr * mot0.R / mot0.km ** 2
        ctl_mot3.Ks = Ks
        ctl_mot3.Umin = -mot0.Umax
        ctl_mot3.Umax = mot0.Umax

        grp = Group(
            "grp",
            inputs={"in0": (1,), "in1": (1,), "in2": (1,), "in3": (1,)},
            snames=["gs0", "gs1", "gs2", "gs3"],
        )
        sys = Quadri(mot0)
        x0 = sys.getInitialStateForOutput("state")
        w0 = np.array([2, -1, 3]) / 2
        x0[10:13] = w0
        sys.setInitialStateForOutput(x0, "state")

        A0 = sys.g
        stp = Rectangular("stp", snames=["r", "p", "y", "A"])
        stp.doors = np.array(
            [
                (10, np.pi / 4, 0, 20),
                (30, np.pi / 4, 0, 40),
                (50, np.pi / 4, 0, 60),
                (-1, A0, A0, 71),
            ]
        )
        ctl = AttPilot("ctlatt", sys, mot0)

        spt = Split(
            name="spt",
            signal_shape=(7,),
            outputs={"u0": [0], "u1": [1], "u2": [2], "u3": [3]},
        )

        sim = Simulation()
        sim.addComputer(mot0)
        sim.addComputer(mot1)
        sim.addComputer(mot2)
        sim.addComputer(mot3)
        sim.addComputer(grp)
        sim.addComputer(stp)
        sim.addComputer(sys)
        sim.addComputer(ctl)
        sim.addComputer(spt)
        sim.addComputer(ctl_mot0)
        sim.addComputer(ctl_mot1)
        sim.addComputer(ctl_mot2)
        sim.addComputer(ctl_mot3)

        sim.connect("stp.setpoint", "ctlatt.setpoint")
        sim.connect("ctlatt.command", "spt.signal")
        sim.connect("spt.u0", "ctlmot0.setpoint")
        sim.connect("spt.u1", "ctlmot1.setpoint")
        sim.connect("spt.u2", "ctlmot2.setpoint")
        sim.connect("spt.u3", "ctlmot3.setpoint")
        sim.connect("ctlmot0.command", "mot0.command")
        sim.connect("ctlmot1.command", "mot1.command")
        sim.connect("ctlmot2.command", "mot2.command")
        sim.connect("ctlmot3.command", "mot3.command")
        sim.connect("mot0.vel", "ctlmot0.estimation")
        sim.connect("mot1.vel", "ctlmot1.estimation")
        sim.connect("mot2.vel", "ctlmot2.estimation")
        sim.connect("mot3.vel", "ctlmot3.estimation")
        sim.connect("mot0.state", "grp.in0")
        sim.connect("mot1.state", "grp.in1")
        sim.connect("mot2.state", "grp.in2")
        sim.connect("mot3.state", "grp.in3")
        sim.connect("grp.grouped", "sys.command")
        sim.connect("sys.state", "ctlatt.estimation")
        sim.connect("sys.euler", "ctlatt.euler")

        tps = np.arange(0, 70, 0.05)
        sim.setOutputLoggerFile(fic="tests/quadri.log")
        sim.simulate(tps, progress_bar=False)

        cls.log = sim.getLogger()

    def test_cmd_att_final(self):
        self.log = TestCmdAtt.log

        r = self.log.getValue("sys_euler_roll")
        p = self.log.getValue("sys_euler_pitch")
        y = self.log.getValue("sys_euler_yaw")
        self.assertAlmostEqual(np.abs(r[-1]), 0, delta=1e-6)
        self.assertAlmostEqual(np.abs(p[-1]), 0, delta=1e-6)
        self.assertAlmostEqual(np.abs(y[-1]), 0, delta=1e-6)

        # fx = self.log.getValue("fx")
        # fy = self.log.getValue("fy")
        # fz = self.log.getValue("fz")
        # self.assertAlmostEqual(np.abs(fx[-1]), 0, delta=1e-9)
        # self.assertAlmostEqual(np.abs(fy[-1]), 0, delta=1e-9)
        # self.assertAlmostEqual(np.abs(fz[-1]), 0, delta=1e-9)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_cmd_att_angles(self):
        self.log = TestCmdAtt.log

        return self.plotVerif(
            "Figure 1",
            [{"var": "deg(roll)"}, {"var": "deg(r_cons)"}],
            [{"var": "deg(pitch)"}, {"var": "deg(p_cons)"}],
            [{"var": "deg(yaw)"}, {"var": "deg(y_cons)"}],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_cmd_att_sval(self):
        self.log = TestCmdAtt.log

        return self.plotVerif(
            "Figure 2",
            [{"var": "s0"}, {"var": "s0_cons"}],
            [{"var": "s1"}, {"var": "s1_cons"}],
            [{"var": "s2"}, {"var": "s2_cons"}],
            [{"var": "s3"}, {"var": "s3_cons"}],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_cmd_att_torques(self):
        self.log = TestCmdAtt.log

        return self.plotVerif(
            "Figure 3", [{"var": "Gr"}], [{"var": "Gp"}], [{"var": "Gy"}]
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_cmd_att_trans(self):
        self.log = TestCmdAtt.log

        return self.plotVerif(
            "Figure 4",
            [{"var": "px"}, {"var": "py"}, {"var": "pz"}],
            [{"var": "vx"}, {"var": "vy"}, {"var": "vz"}],
            [{"var": "fx"}, {"var": "fy"}, {"var": "fz"}],
        )


if __name__ == "__main__":
    # unittest.main()

    # a = TestQuad()
    # a.test_motor()
    # a.test_quad()

    TestCmdAtt.setUpClass()
    a = TestCmdAtt()
    a.setUp()
    a.test_cmd_att_final()

    plt.show()