import os
import sys
import unittest
from typing import Iterable

import numpy as np
import scipy.linalg as lin
from matplotlib import pyplot as plt
import pytest

from blocksim.blocks.System import LTISystem
from blocksim.blocks.Controller import LQRegulator, AntiWindupPIDController
from blocksim.Simulation import Simulation
from blocksim.blocks.SetPoint import Step
from blocksim.blocks.Route import Group, Split


sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

# from Quadcopter.Quadri import Quadri
# from Quadcopter.AttPilot import AttPilot
# from Quadcopter.Motor import Motor
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
        self.sys.A = np.zeros((6, 6))
        self.sys.A[0:3, 3:6] = np.eye(3)
        self.sys.B = np.zeros((6, 3))
        self.sys.B[3:6, :] = np.eye(3)
        self.sys.m = 0.458
        self.sys.g = 9.81

        self.splt = Split(
            name="split",
            signal_shape=(7,),
            snames=[
                "fx",
                "fy",
                "fz",
            ],
            selected_input=[0, 1, 2],
        )

        self.lqr = LQRegulator(
            "lqr", shape_setpoint=(4,), shape_estimation=(3,), snames=["fx", "fy", "fz"]
        )
        self.lqr.C = np.zeros((3, 6))
        self.lqr.C[:, 0:3] = np.eye(3)
        self.lqr.D = np.zeros((3, 3))
        self.lqr.computeGain(np.eye(6), np.eye(3) * 5, self.sys)

        self.ctl = VTOLPilot(self.sys, self.lqr, complex_quad=False)

        self.stp = Step(
            "stp",
            cons=np.array([1, 1, 1, 0]),
            snames=["x_cons", "y_cons", "z_cons", "psi_cons"],
        )
        print(self.stp)
        print(self.sys)
        print(self.ctl)
        print(self.splt)

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


# class TestPVTOLComplex(TestBase):
#     @classmethod
#     def setUpClass(cls):
#         """get_some_resource() is slow, to avoid calling it for each test use setUpClass()
#         and store the result as class variable
#         """
#         super(TestPVTOLComplex, cls).setUpClass()

#         mot0 = Motor(0)
#         mot1 = Motor(1)
#         mot2 = Motor(2)
#         mot3 = Motor(3)

#         tau = 50e-3
#         Ks = 0.0

#         ctl_mot0 = AntiWindupPIDController(
#             "ctl_mot0",
#             name_of_outputs=["umot0"],
#             name_of_states=["state_umot0", "int_x0", "corr0"],
#         )
#         ctl_mot0.D = 0.0
#         ctl_mot0.I = mot0.km / tau
#         ctl_mot0.P = ctl_mot0.I * mot0.Jr * mot0.R / mot0.km ** 2
#         ctl_mot0.Ks = Ks
#         ctl_mot0.Umin = -mot0.Umax
#         ctl_mot0.Umax = mot0.Umax

#         ctl_mot1 = AntiWindupPIDController(
#             "ctl_mot1",
#             name_of_outputs=["umot1"],
#             name_of_states=["state_umot1", "int_x1", "corr1"],
#         )
#         ctl_mot1.D = 0.0
#         ctl_mot1.I = mot0.km / tau
#         ctl_mot1.P = ctl_mot1.I * mot0.Jr * mot0.R / mot0.km ** 2
#         ctl_mot1.Ks = Ks
#         ctl_mot1.Umin = -mot0.Umax
#         ctl_mot1.Umax = mot0.Umax

#         ctl_mot2 = AntiWindupPIDController(
#             "ctl_mot2",
#             name_of_outputs=["umot2"],
#             name_of_states=["state_umot2", "int_x2", "corr2"],
#         )
#         ctl_mot2.D = 0.0
#         ctl_mot2.I = mot0.km / tau
#         ctl_mot2.P = ctl_mot2.I * mot0.Jr * mot0.R / mot0.km ** 2
#         ctl_mot2.Ks = Ks
#         ctl_mot2.Umin = -mot0.Umax
#         ctl_mot2.Umax = mot0.Umax

#         ctl_mot3 = AntiWindupPIDController(
#             "ctl_mot3",
#             name_of_outputs=["umot3"],
#             name_of_states=["state_umot3", "int_x3", "corr3"],
#         )
#         ctl_mot3.D = 0.0
#         ctl_mot3.I = mot0.km / tau
#         ctl_mot3.P = ctl_mot3.I * mot0.Jr * mot0.R / mot0.km ** 2
#         ctl_mot3.Ks = Ks
#         ctl_mot3.Umin = -mot0.Umax
#         ctl_mot3.Umax = mot0.Umax

#         grp = Group(
#             "grp",
#             name_of_outputs=["gs0", "gs1", "gs2", "gs3"],
#             name_of_inputs=["s0", "s1", "s2", "s3"],
#         )

#         vtol_sys = LTISystem(
#             "sys",
#             name_of_outputs=["x", "y", "z"],
#             name_of_states=[
#                 "state_x",
#                 "state_y",
#                 "state_z",
#                 "state_vx",
#                 "state_vy",
#                 "state_vz",
#             ],
#         )
#         vtol_sys.setInitialState((np.random.rand(6) - 0.5) * 20)
#         vtol_sys.A = np.zeros((6, 6))
#         vtol_sys.A[0:3, 3:6] = np.eye(3)
#         vtol_sys.B = np.zeros((6, 3))
#         vtol_sys.B[3:6, :] = np.eye(3)
#         vtol_sys.C = np.zeros((3, 6))
#         vtol_sys.C[:, 0:3] = np.eye(3)
#         vtol_sys.D = np.zeros((3, 3))
#         vtol_sys.createParameter("m", 0.458)
#         vtol_sys.createParameter("g", 9.81)

#         lqr = LQRegulator("lqr", name_of_outputs=["fx", "fy", "fz"])
#         lqr.computeGain(np.eye(6), np.eye(3) * 5, vtol_sys)

#         att_sys = Quadri(mot0)
#         x0 = att_sys.getInitialState()
#         x0[:6] = np.array([0, 0, 0, 2.349619, -2.409138, 4.907362])
#         w0 = np.array([2, -1, 3]) / 2
#         x0[10:13] = w0
#         att_sys.setInitialState(x0)
#         stp = Step(
#             "stp",
#             cons=np.array([0, 0, 0, 0]),
#             name_of_outputs=["x_cons", "y_cons", "z_cons", "psi_cons"],
#         )
#         ctl_att = AttPilot(att_sys, mot0)
#         ctl_vtol = VTOLPilot(vtol_sys, lqr, complex_quad=True)

#         sim = Simulation()
#         sim.addElement(mot0)
#         sim.addElement(mot1)
#         sim.addElement(mot2)
#         sim.addElement(mot3)
#         sim.addElement(grp)
#         sim.addElement(stp)
#         sim.addElement(att_sys)
#         sim.addElement(ctl_vtol)
#         sim.addElement(ctl_att)
#         sim.addElement(ctl_mot0)
#         sim.addElement(ctl_mot1)
#         sim.addElement(ctl_mot2)
#         sim.addElement(ctl_mot3)

#         sim.linkElements(
#             src=stp, dst=ctl_vtol, src_data_name="state", dst_input_name="setpoint"
#         )
#         sim.linkElements(
#             src=ctl_vtol, dst=ctl_att, src_data_name="output", dst_input_name="setpoint"
#         )
#         sim.linkElements(
#             src=ctl_att,
#             dst=ctl_mot0,
#             src_data_name="output[0]",
#             dst_input_name="setpoint",
#         )
#         sim.linkElements(
#             src=ctl_att,
#             dst=ctl_mot1,
#             src_data_name="output[1]",
#             dst_input_name="setpoint",
#         )
#         sim.linkElements(
#             src=ctl_att,
#             dst=ctl_mot2,
#             src_data_name="output[2]",
#             dst_input_name="setpoint",
#         )
#         sim.linkElements(
#             src=ctl_att,
#             dst=ctl_mot3,
#             src_data_name="output[3]",
#             dst_input_name="setpoint",
#         )
#         sim.linkElements(
#             src=ctl_mot0, dst=mot0, src_data_name="output", dst_input_name="command"
#         )
#         sim.linkElements(
#             src=ctl_mot1, dst=mot1, src_data_name="output", dst_input_name="command"
#         )
#         sim.linkElements(
#             src=ctl_mot2, dst=mot2, src_data_name="output", dst_input_name="command"
#         )
#         sim.linkElements(
#             src=ctl_mot3, dst=mot3, src_data_name="output", dst_input_name="command"
#         )
#         sim.linkElements(
#             src=mot0, dst=ctl_mot0, src_data_name="output", dst_input_name="estimation"
#         )
#         sim.linkElements(
#             src=mot1, dst=ctl_mot1, src_data_name="output", dst_input_name="estimation"
#         )
#         sim.linkElements(
#             src=mot2, dst=ctl_mot2, src_data_name="output", dst_input_name="estimation"
#         )
#         sim.linkElements(
#             src=mot3, dst=ctl_mot3, src_data_name="output", dst_input_name="estimation"
#         )
#         sim.linkElements(src=mot0, dst=grp, src_data_name="state", dst_input_name="s0")
#         sim.linkElements(src=mot1, dst=grp, src_data_name="state", dst_input_name="s1")
#         sim.linkElements(src=mot2, dst=grp, src_data_name="state", dst_input_name="s2")
#         sim.linkElements(src=mot3, dst=grp, src_data_name="state", dst_input_name="s3")
#         sim.linkElements(
#             src=grp, dst=att_sys, src_data_name="output", dst_input_name="command"
#         )
#         sim.linkElements(
#             src=att_sys,
#             dst=ctl_att,
#             src_data_name="output",
#             dst_input_name="estimation",
#         )
#         sim.linkElements(
#             src=att_sys,
#             dst=ctl_vtol,
#             src_data_name="output",
#             dst_input_name="estimation",
#         )

#         tps = np.arange(0, 20, 0.01)
#         sim.setOutputLoggerFile(fic="tests/quadri.log")
#         sim.simulate(tps, progress_bar=False)

#         cls.log = sim.getLogger()

#     def test_quad_complexe_final(self):
#         self.log = TestPVTOLComplex.log

#         t = self.log.getValue("t")
#         iok = np.where(t > 15)[0]

#         px = self.log.getValue("px")[iok]
#         py = self.log.getValue("py")[iok]
#         pz = self.log.getValue("pz")[iok]
#         yaw = self.log.getValue("yaw")[iok]

#         self.assertAlmostEqual(np.max(np.abs(px)), 0, delta=0.1)
#         self.assertAlmostEqual(np.max(np.abs(py)), 0, delta=0.1)
#         self.assertAlmostEqual(np.max(np.abs(pz)), 0, delta=0.1)
#         self.assertAlmostEqual(np.max(np.abs(yaw)), 0, delta=1e-5)

#     @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
#     def test_quad_complexe_yaw(self):
#         self.log = TestPVTOLComplex.log

#         return self.plotVerif(
#             "Figure 1",
#             [{"var": "px"}, {"var": "py"}, {"var": "pz"}],
#             [{"var": "deg(yaw)"}],
#         )

#     @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
#     def test_quad_complexe_sval(self):
#         self.log = TestPVTOLComplex.log

#         return self.plotVerif(
#             "Figure 2", [{"var": "s0"}, {"var": "s1"}, {"var": "s2"}, {"var": "s3"}]
#         )

#     @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
#     def test_quad_complexe_att(self):
#         self.log = TestPVTOLComplex.log

#         return self.plotVerif(
#             "Figure 3",
#             [
#                 {"var": "deg(roll_cons)"},
#                 {"var": "deg(pitch_cons)"},
#                 {"var": "deg(yaw_cons)"},
#             ],
#         )


if __name__ == "__main__":
    # unittest.main()

    a = TestPVTOL()
    a.setUp()
    a.test_quad_simplified()

    plt.show()
