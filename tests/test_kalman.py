import sys
import os
import unittest
from collections import OrderedDict

import numpy as np
from numpy import cos, sin, sqrt, exp, pi
from matplotlib import pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.core.Node import Frame
from blocksim.blocks.SetPoint import Step
from blocksim.blocks.System import LTISystem
from blocksim.blocks.Controller import PIDController, LQRegulator
from blocksim.blocks.Estimator import SteadyStateKalmanFilter, TimeInvariantKalmanFilter
from blocksim.blocks.Sensors import LinearSensors
from blocksim.blocks.Route import Split
from blocksim.Simulation import Simulation


class TestKalman(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_ss_kal(self):
        m = 1.0  # Mass
        k = 40.0  # Spring rate
        f = 5
        dt = 0.05

        sys = LTISystem("sys", shape_command=(1,), snames_state=["x", "v"])
        sys.matA = np.array([[0, 1], [-k / m, -f / m]])
        sys.matB = np.array([[0, 1 / m]]).T
        sys.setInitialStateForOutput(np.array([-1.0, 0.0]), "state")

        kal = SteadyStateKalmanFilter(
            "kal",
            dt=dt,
            shape_cmd=(1,),
            shape_meas=(1,),
            snames_state=["x", "v"],
            snames_output=["x"],
        )
        kal.matA = sys.matA
        kal.matB = sys.matB
        kal.matC = np.array([[1, 0]])
        kal.matD = np.zeros((1, 1))
        kal.matQ = np.eye(2) / 10000
        kal.matR = np.eye(1) / 100

        cpt = LinearSensors("cpt", shape_state=(2,), shape_command=(1,), snames=["x"])
        cpt.matC = np.array([1, 0])
        cpt.matD = np.zeros((1, 1))
        cpt.setCovariance(np.eye(1) / 200)
        cpt.setMean(np.zeros(1))

        a = 8
        P = -k + 3 * a ** 2 * m
        I = a ** 3 * m
        D = 3 * a * m
        ctl = PIDController(
            "ctl", shape_estimation=(2,), snames=["u"], coeffs=(P, I, D)
        )

        stp = Step(name="stp", snames=["c"], cons=np.array([1]))

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(ctl)
        sim.addComputer(cpt)
        sim.addComputer(kal)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "ctl.setpoint")
        sim.connect("ctl.command", "sys.command")
        sim.connect("sys.state", "cpt.state")
        sim.connect("ctl.command", "cpt.command")
        sim.connect("cpt.measurement", "kal.measurement")
        sim.connect("ctl.command", "kal.command")
        sim.connect("kal.state", "ctl.estimation")

        tps = np.arange(0, 4, dt)
        frame = sim.simulate(tps, progress_bar=True)

        self.log = sim.getLogger()

        return self.plotVerif(
            "test_ss_kal",
            [
                {"var": "sys_state_x"},
                {"var": "stp_setpoint_c"},
                {"var": "kal_state_x"},
                {"var": "cpt_measurement_x", "linestyle": "", "marker": "+"},
            ],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_ti_kal(self):
        m = 1.0  # Mass
        k = 40.0  # Spring rate
        f = 5
        dt = 0.05
        bias = 0.5

        sys = LTISystem("sys", shape_command=(1,), snames_state=["x", "v"])
        sys.matA = np.array([[0, 1], [-k / m, -f / m]])
        sys.matB = np.array([[0, 1 / m]]).T
        sys.setInitialStateForOutput(np.array([-1.0, 0.0]), "state")

        kal = TimeInvariantKalmanFilter(
            "kal",
            shape_cmd=(1,),
            shape_meas=(1,),
            snames_state=["x", "v", "b"],
            snames_output=["x"],
        )
        kal.matA = np.zeros((3, 3))
        kal.matA[:2, :2] = sys.matA
        kal.matB = np.zeros((3, 1))
        kal.matB[:2] = sys.matB
        kal.matC = np.array([[1, 0, 1]])
        kal.matD = np.zeros((1, 1))
        kal.matQ = np.eye(3) / 10000
        kal.matR = np.eye(1) / 100

        cpt = LinearSensors("cpt", shape_state=(2,), shape_command=(1,), snames=["x"])
        cpt.matC = np.array([1, 0])
        cpt.matD = np.zeros((1, 1))
        cpt.setCovariance(np.eye(1) / 200)
        cpt.setMean(np.array([bias]))

        ctl = LQRegulator(
            "ctl", shape_setpoint=(1,), shape_estimation=(2,), snames=["u"]
        )
        ctl.matA = sys.matA
        ctl.matB = sys.matB
        ctl.matC = kal.matC[:, :2]
        ctl.matD = kal.matD
        ctl.matQ = np.eye(2) / 10000
        ctl.matR = np.eye(1) / 100
        ctl.computeGain()

        stp = Step(name="stp", snames=["c"], cons=np.array([1]))

        spt_otp = OrderedDict()
        spt_otp["split"] = [0, 1]
        split = Split("split", signal_shape=(3,), outputs=spt_otp)

        sim = Simulation()
        sim.addComputer(sys)
        sim.addComputer(ctl)
        sim.addComputer(cpt)
        sim.addComputer(kal)
        sim.addComputer(split)
        sim.addComputer(stp)

        sim.connect("stp.setpoint", "ctl.setpoint")
        sim.connect("ctl.command", "sys.command")
        sim.connect("sys.state", "cpt.state")
        sim.connect("ctl.command", "cpt.command")
        sim.connect("cpt.measurement", "kal.measurement")
        sim.connect("ctl.command", "kal.command")
        sim.connect("kal.state", "split.signal")
        sim.connect("split.split", "ctl.estimation")

        tps = np.arange(0, 4, dt)
        frame = sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()
        kal_bias = self.log.getValue("kal_state_b")
        self.assertAlmostEqual(np.abs(kal_bias[-1] - bias), 0, delta=5e-3)

        return self.plotVerif(
            "test_ss_kal",
            [
                {"var": "sys_state_x"},
                {"var": "stp_setpoint_c"},
                {"var": "kal_state_x"},
                {"var": "cpt_measurement_x", "linestyle": "", "marker": "+"},
            ],
        )


if __name__ == "__main__":
    # unittest.main()

    a = TestKalman()
    # a.test_ss_kal()
    a.test_ti_kal()

    plt.show()
