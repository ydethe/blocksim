import os
import sys
import unittest
from typing import Iterable
from collections import OrderedDict

import numpy as np
import scipy.linalg as lin
from matplotlib import pyplot as plt
import pytest

from blocksim.control.System import LTISystem
from blocksim.control.Controller import LQRegulator, AntiWindupPIDController
from blocksim.Simulation import Simulation
from blocksim.control.SetPoint import Step
from blocksim.control.Route import Group, Split


sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from Quadcopter.Quadri import Quadri
from Quadcopter.AttPilot import AttPilot
from Quadcopter.Motor import Motor
from Quadcopter.VTOLPilot import VTOLPilot


class TestPVTOL(TestBase):
    def setUp(self):
        super().setUp()

        self.sys = LTISystem(
            "sys",
            shape_command=(3,),
            snames_state=[
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
            ],
        )
        self.sys.setInitialStateForOutput((np.random.rand(6) - 0.5) * 20, "state")
        self.sys.matA = np.zeros((6, 6))
        self.sys.matA[0:3, 3:6] = np.eye(3)
        self.sys.matB = np.zeros((6, 3))
        self.sys.matB[3:6, :] = np.eye(3)
        self.sys.createParameter("m", value=0.458)
        self.sys.createParameter("g", value=9.81)

        spt_otp = OrderedDict()
        spt_otp["split"] = [0, 1, 2]
        self.splt = Split(
            name="split",
            signal_shape=(7,),
            outputs=spt_otp,
        )

        self.lqr = LQRegulator(
            "lqr", shape_setpoint=(4,), shape_estimation=(3,), snames=["fx", "fy", "fz"]
        )
        self.lqr.matA = self.sys.matA
        self.lqr.matB = self.sys.matB
        self.lqr.matC = np.zeros((3, 6))
        self.lqr.matC[:, 0:3] = np.eye(3)
        self.lqr.matD = np.zeros((3, 3))
        self.lqr.matQ = np.eye(6)
        self.lqr.matR = np.eye(3) * 5
        self.lqr.computeGain()

        self.ctl = VTOLPilot(self.sys, self.lqr, complex_quad=False)

        self.stp = Step(
            "stp",
            cons=np.array([1, 1, 1, 0]),
            snames=["x_cons", "y_cons", "z_cons", "psi_cons"],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_quad_simplified(self):
        sim = Simulation()
        sim.addComputer(self.stp)
        sim.addComputer(self.sys)
        sim.addComputer(self.ctl)
        sim.addComputer(self.splt)

        sim.connect("stp.setpoint", "ctlvtol.setpoint")
        sim.connect("sys.state", "ctlvtol.estimation")
        sim.connect("ctlvtol.command", "split.signal")
        sim.connect("split.split", "sys.command")

        tps = np.arange(0, 10, 0.01)
        sim.setOutputLoggerFile(fic="tests/quadri.log")
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()
        return self.plotVerif(
            "Figure 1",
            [{"var": "sys_state_x"}, {"var": "sys_state_y"}, {"var": "sys_state_z"}],
        )


class TestPVTOLComplex(TestBase):
    @classmethod
    def setUpClass(cls):
        """get_some_resource() is slow, to avoid calling it for each test use setUpClass()
        and store the result as class variable
        """
        super(TestPVTOLComplex, cls).setUpClass()

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

        grp_inp = OrderedDict()
        grp_inp["in0"] = (1,)
        grp_inp["in1"] = (1,)
        grp_inp["in2"] = (1,)
        grp_inp["in3"] = (1,)
        grp = Group(
            "grp",
            inputs=grp_inp,
            snames=["gs0", "gs1", "gs2", "gs3"],
        )

        vtol_sys = LTISystem(
            "sys",
            shape_command=(3,),
            snames_state=[
                "state_x",
                "state_y",
                "state_z",
                "state_vx",
                "state_vy",
                "state_vz",
            ],
        )
        vtol_sys.setInitialStateForOutput((np.random.rand(6) - 0.5) * 20, "state")
        vtol_sys.matA = np.zeros((6, 6))
        vtol_sys.matA[0:3, 3:6] = np.eye(3)
        vtol_sys.matB = np.zeros((6, 3))
        vtol_sys.matB[3:6, :] = np.eye(3)
        vtol_sys.createParameter("m", 0.458)
        vtol_sys.createParameter("g", 9.81)

        lqr = LQRegulator(
            "lqr", shape_setpoint=(4,), shape_estimation=(6,), snames=["fx", "fy", "fz"]
        )
        lqr.matQ = np.eye(6)
        lqr.matR = np.eye(3) * 5
        lqr.matA = vtol_sys.matA
        lqr.matB = vtol_sys.matB
        lqr.matC = np.zeros((3, 6))
        lqr.matC[:, 0:3] = np.eye(3)
        lqr.matD = np.zeros((3, 3))
        lqr.computeGain()

        att_sys = Quadri(mot0)
        x0 = att_sys.getInitialStateForOutput("state")
        x0[:6] = np.array([0, 0, 0, 2.349619, -2.409138, 4.907362])
        w0 = np.array([2, -1, 3]) / 2
        x0[10:13] = w0
        att_sys.setInitialStateForOutput(x0, "state")
        stp = Step(
            "stp",
            cons=np.array([0, 0, 0, 0]),
            snames=["x_cons", "y_cons", "z_cons", "psi_cons"],
        )
        ctlatt = AttPilot("ctlatt", att_sys, mot0)
        ctlvtol = VTOLPilot(vtol_sys, lqr, complex_quad=True)

        spt_otp = OrderedDict()
        spt_otp["u0"] = (0,)
        spt_otp["u1"] = (1,)
        spt_otp["u2"] = (2,)
        spt_otp["u3"] = (3,)
        spt = Split(
            name="spt",
            signal_shape=(7,),
            outputs=spt_otp,
        )

        vspt_otp = OrderedDict()
        vspt_otp["att"] = [3, 4, 5, 6]
        vtolspt = Split(
            name="vtolspt",
            signal_shape=(7,),
            outputs=vspt_otp,
        )

        wspt_otp = OrderedDict()
        wspt_otp["pv"] = list(range(6))
        wtolspt = Split(
            name="wtolspt",
            signal_shape=(13,),
            outputs=wspt_otp,
        )

        sim = Simulation()
        sim.addComputer(mot0)
        sim.addComputer(mot1)
        sim.addComputer(mot2)
        sim.addComputer(mot3)
        sim.addComputer(grp)
        sim.addComputer(stp)
        sim.addComputer(att_sys)
        sim.addComputer(ctlvtol)
        sim.addComputer(ctlatt)
        sim.addComputer(spt)
        sim.addComputer(vtolspt)
        sim.addComputer(wtolspt)
        sim.addComputer(ctl_mot0)
        sim.addComputer(ctl_mot1)
        sim.addComputer(ctl_mot2)
        sim.addComputer(ctl_mot3)

        sim.connect("stp.setpoint", "ctlvtol.setpoint")
        sim.connect("ctlatt.command", "spt.signal")
        sim.connect("ctlvtol.command", "vtolspt.signal")
        sim.connect("vtolspt.att", "ctlatt.setpoint")
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
        sim.connect("sys.state", "wtolspt.signal")
        sim.connect("wtolspt.pv", "ctlvtol.estimation")

        tps = np.arange(0, 20, 0.01)
        sim.setOutputLoggerFile(fic="tests/quadri.log")
        sim.simulate(tps, progress_bar=False)

        cls.log = sim.getLogger()

    def test_quad_complexe_final(self):
        self.log = TestPVTOLComplex.log

        t = self.log.getValue("t")
        iok = np.where(t > 15)[0]

        px = self.log.getValue("sys_state_px")[iok]
        py = self.log.getValue("sys_state_px")[iok]
        pz = self.log.getValue("sys_state_px")[iok]
        yaw = self.log.getValue("sys_euler_yaw")[iok]

        self.assertAlmostEqual(np.max(np.abs(px)), 0, delta=0.1)
        self.assertAlmostEqual(np.max(np.abs(py)), 0, delta=0.1)
        self.assertAlmostEqual(np.max(np.abs(pz)), 0, delta=0.1)
        self.assertAlmostEqual(np.max(np.abs(yaw)), 0, delta=1e-5)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_quad_complexe_yaw(self):
        self.log = TestPVTOLComplex.log

        return self.plotVerif(
            "Figure 1",
            [{"var": "sys_state_px"}, {"var": "sys_state_py"}, {"var": "sys_state_pz"}],
            [{"var": "deg(sys_euler_yaw)"}],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_quad_complexe_sval(self):
        self.log = TestPVTOLComplex.log

        return self.plotVerif(
            "Figure 2",
            [
                {"var": "mot0_vel_s"},
                {"var": "mot1_vel_s"},
                {"var": "mot2_vel_s"},
                {"var": "mot3_vel_s"},
            ],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_quad_complexe_att(self):
        self.log = TestPVTOLComplex.log

        return self.plotVerif(
            "Figure 3",
            [
                {"var": "deg(ctlvtol_command_roll)"},
                {"var": "deg(ctlvtol_command_pitch)"},
                {"var": "deg(ctlvtol_command_yaw)"},
            ],
        )


if __name__ == "__main__":
    unittest.main()

    # a = TestPVTOLComplex()
    # TestPVTOLComplex.setUpClass()
    # a.setUp()
    # a.test_quad_complexe_att()

    # plt.show()
