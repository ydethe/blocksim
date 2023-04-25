import unittest

import numpy as np
from scipy import linalg as lin
import pytest

from blocksim.control.SetPoint import Step
from blocksim.control.System import LTISystem
from blocksim.control.Controller import (
    PIDController,
    LQRegulator,
    AntiWindupPIDController,
)
from blocksim.control.Estimator import (
    SteadyStateKalmanFilter,
    TimeInvariantKalmanFilter,
)
from blocksim.control.Sensors import LinearSensors, ProportionalSensors
from blocksim.control.Route import Split
from blocksim.Simulation import Simulation


from blocksim.testing import TestBase


class TestKalman(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_awc_kal(self):
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
            snames_state=["x", "v"],
            snames_output=["x"],
        )
        kal.matA = sys.matA
        kal.matB = sys.matB
        kal.matC = np.array([[1, 0]])
        kal.matD = np.zeros((1, 1))
        kal.matQ = np.eye(2) / 10000
        kal.matR = np.eye(1) / 100

        cpt = ProportionalSensors("cpt", shape_state=(2,), snames=["x"])
        cpt.matC = np.array([1, 0])
        cpt.setCovariance(np.eye(1) / 200)
        cpt.setMean(np.zeros(1))

        err_cov = lin.norm(cpt.getCovariance() - np.eye(1) / 200)
        err_mean = lin.norm(cpt.getMean() - np.zeros(1))
        self.assertAlmostEqual(err_cov, 0, delta=1e-9)
        self.assertAlmostEqual(err_mean, 0, delta=1e-9)

        a = 8
        Kprop = -k + 3 * a**2 * m
        Kinteg = a**3 * m
        Kderiv = 3 * a * m
        Umin = -100
        Umax = 100
        Ks = 0
        ctl = AntiWindupPIDController(
            "ctl",
            shape_estimation=(2,),
            snames=["u"],
            coeffs=(Kprop, Kinteg, Kderiv, Umin, Umax, Ks),
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
        sim.connect("cpt.measurement", "kal.measurement")
        sim.connect("ctl.command", "kal.command")
        sim.connect("kal.state", "ctl.estimation")

        tps = np.arange(0, 4, dt)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()
        u = self.log.getRawValue("ctl_command_u")
        self.assertGreaterEqual(np.min(u), Umin)
        self.assertLessEqual(np.max(u), Umax)

        fig = self.plotVerif(
            "test_awc_kal",
            [
                {"var": "sys_state_x"},
                {"var": "stp_setpoint_c"},
                {"var": "kal_state_x"},
                {"var": "cpt_measurement_x", "linestyle": "", "marker": "+"},
            ],
        )
        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
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
            name="kal",
            dt=dt,
            shape_cmd=(1,),
            snames_state=["x", "v"],
            snames_output=["x"],
        )
        kal.matA = sys.matA
        kal.matB = sys.matB
        kal.matC = np.array([[1, 0]])
        kal.matD = np.zeros((1, 1))
        kal.matQ = np.eye(2) / 10000
        kal.matR = np.eye(1) / 100

        cpt = ProportionalSensors("cpt", shape_state=(2,), snames=["x"])
        cpt.matC = np.array([1, 0])
        cpt.setCovariance(np.eye(1) / 200)
        cpt.setMean(np.zeros(1))

        err_cov = lin.norm(cpt.getCovariance() - np.eye(1) / 200)
        err_mean = lin.norm(cpt.getMean() - np.zeros(1))
        self.assertAlmostEqual(err_cov, 0, delta=1e-9)
        self.assertAlmostEqual(err_mean, 0, delta=1e-9)

        a = 8
        Kprop = -k + 3 * a**2 * m
        Kinteg = a**3 * m
        Kderiv = 3 * a * m
        ctl = PIDController(
            "ctl", shape_estimation=(2,), snames=["u"], coeffs=(Kprop, Kinteg, Kderiv)
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
        sim.connect("cpt.measurement", "kal.measurement")
        sim.connect("ctl.command", "kal.command")
        sim.connect("kal.state", "ctl.estimation")

        tps = np.arange(0, 4, dt)
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        fig = self.plotVerif(
            "test_ss_kal",
            [
                {"var": "sys_state_x"},
                {"var": "stp_setpoint_c"},
                {"var": "kal_state_x"},
                {"var": "cpt_measurement_x", "linestyle": "", "marker": "+"},
            ],
        )
        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
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

        ctl = LQRegulator("ctl", shape_setpoint=(1,), shape_estimation=(2,), snames=["u"])
        ctl.matA = sys.matA
        ctl.matB = sys.matB
        ctl.matC = kal.matC[:, :2]
        ctl.matD = kal.matD
        ctl.matQ = np.eye(2) / 10000
        ctl.matR = np.eye(1) / 100

        stp = Step(name="stp", snames=["c"], cons=np.array([1]))

        spt_otp = dict()
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
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()
        kal_bias = self.log.getValue("kal_state_b")
        self.assertAlmostEqual(kal_bias[-1], bias, delta=2e-2)

        fig = self.plotVerif(
            "test_ss_kal",
            [
                {"var": "sys_state_x"},
                {"var": "stp_setpoint_c"},
                {"var": "kal_state_x"},
                {"var": "cpt_measurement_x", "linestyle": "", "marker": "+"},
            ],
        )
        return fig.render()


if __name__ == "__main__":
    unittest.main()
    exit(0)

    from blocksim.graphics import showFigures

    a = TestKalman()
    # a.test_ss_kal()
    # a.test_ti_kal()
    a.test_awc_kal()

    showFigures()
