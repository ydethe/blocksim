import os
import sys
import unittest

import numpy as np
import scipy.linalg as lin
from matplotlib import pyplot as plt
import pytest

from blocksim.Simulation import Simulation
from blocksim.blocks.Controller import PIDController
from blocksim.blocks.System import ASystem
from blocksim.blocks.Sensors import ASensors
from blocksim.blocks.SetPoint import Step
from blocksim.blocks.Estimator import MadgwickFilter, MahonyFilter
from blocksim.core.Node import AComputer
from blocksim.utils import deg, rad


sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class IMU(ASensors):
    __slots__ = []

    def __init__(self):
        moy = np.zeros(9)
        moy[0] = 0.5 * np.pi / 180
        moy[1] = -1.0 * np.pi / 180
        moy[2] = 1.5 * np.pi / 180
        cov = np.diag(3 * [np.pi / 180] + 3 * [1e-3 * 9.81] + 3 * [1.0e-6])
        ASensors.__init__(
            self,
            name="imu",
            shape_state=(9,),
            snames=[
                "gx",
                "gy",
                "gz",
                "ax",
                "ay",
                "az",
                "mx",
                "my",
                "mz",
            ],
        )
        self.setCovariance(cov)
        self.setMean(moy)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        measurement: np.array,
        state: np.array,
    ) -> dict:
        outputs = {}
        outputs["measurement"] = state.copy()
        return outputs


class TSystem(ASystem):
    __slots__ = []

    def __init__(self):
        ASystem.__init__(
            self,
            name="sys",
            shape_command=(9,),
            snames_state=["gx", "gy", "gz", "ax", "ay", "az", "mx", "my", "mz"],
            method="vode",
        )
        self.createParameter(name="w", value=0.0)

    def transition(self, t, x, u):
        gx, gy, gz, ax, ay, az, mx, my, mz = x
        dxdt = np.zeros(9)
        dxdt[3:6] = np.cross(np.array([ax, ay, az]), self.w)
        dxdt[6:9] = np.cross(np.array([mx, my, mz]), self.w)
        return dxdt


class TestIMU(TestBase):
    def setUp(self):
        super().setUp()

        self.dt = 1e-2

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_madgwick_pitch(self):
        np.random.seed(134697)

        ctrl = Step(name="ctrl", snames=["u%i" % i for i in range(9)], cons=np.zeros(9))

        angle_ini = -60 * np.pi / 180.0
        wangle = 10.0 * np.pi / 180.0
        sys = TSystem()
        x0 = np.zeros(9)

        # ==================================================
        # Rotation autour de l'axe de tangage
        # ==================================================
        sys.w = np.array([0.0, wangle, 0.0])

        x0[:3] = sys.w
        x0[3:6] = np.array([-np.sin(angle_ini), 0.0, np.cos(angle_ini)])
        x0[6:9] = np.array([0.0, 1.0, 0.0])

        sys.setInitialStateForOutput(x0, "state")

        c = IMU()

        est = MadgwickFilter("madg", beta=2.0)
        est.setMagnetometerCalibration(
            offset=np.arange(3), softiron_matrix=np.eye(3) / 2
        )
        b, m = est.getMagnetometerCalibration()
        self.assertAlmostEqual(lin.norm(b - np.arange(3)), 0.0, delta=1.0e-6)
        self.assertAlmostEqual(lin.norm(m - np.eye(3) / 2), 0.0, delta=1.0e-6)
        est.setMagnetometerCalibration(offset=np.zeros(3), softiron_matrix=np.eye(3))

        tfin = -2 * angle_ini / wangle
        tps = np.arange(0.0, tfin, self.dt)

        sim = Simulation()

        sim.addComputer(ctrl)
        sim.addComputer(sys)
        sim.addComputer(c)
        sim.addComputer(est)

        sim.connect("ctrl.setpoint", "sys.command")
        sim.connect("sys.state", "imu.state")
        sim.connect("imu.measurement", "madg.measurement")

        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        iok = np.where(tps > 2.0)[0]
        w = angle_ini + tps * wangle
        err_a = np.max(np.abs(self.log.getValue("madg_euler_pitch")[iok] - w[iok]))

        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_a, 0.0, delta=0.11)

        return self.plotVerif(
            "Figure 1",
            [
                {"var": "deg(madg_euler_roll)", "label": "FilteredRoll"},
                {"var": "deg(madg_euler_pitch)", "label": "FilteredPitch"},
                {"var": "deg(madg_euler_yaw)", "label": "FilteredYaw"},
                {
                    "var": deg(w),
                    "label": "Simu",
                    "color": "black",
                    "linestyle": "--",
                },
            ],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_madgwick_yaw(self):
        np.random.seed(134697)

        ctrl = Step(name="ctrl", snames=["u%i" % i for i in range(9)], cons=np.zeros(9))

        angle_ini = -60 * np.pi / 180.0
        wangle = 10.0 * np.pi / 180.0
        sys = TSystem()
        x0 = np.zeros(9)

        # ==================================================
        # Rotation autour de l'axe de lacet
        # ==================================================
        sys.w = np.array([0.0, 0.0, wangle])

        x0[:3] = sys.w
        x0[3:6] = np.array([0.0, 0.0, 1.0])
        x0[6:9] = np.array([np.cos(angle_ini), -np.sin(angle_ini), 0.0])

        sys.setInitialStateForOutput(x0, "state")

        c = IMU()

        est = MadgwickFilter("madg", beta=2.0)

        tfin = -2 * angle_ini / wangle
        tps = np.arange(0.0, tfin, self.dt)

        sim = Simulation()

        sim.addComputer(ctrl)
        sim.addComputer(sys)
        sim.addComputer(c)
        sim.addComputer(est)

        sim.connect("ctrl.setpoint", "sys.command")
        sim.connect("sys.state", "imu.state")
        sim.connect("imu.measurement", "madg.measurement")

        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        iok = np.where(tps > 2.0)[0]
        w = angle_ini + tps * wangle
        err_a = np.max(np.abs(self.log.getValue("madg_euler_yaw")[iok] - w[iok]))

        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_a, 0.0, delta=0.06)

        return self.plotVerif(
            "Figure 1",
            [
                {"var": "deg(madg_euler_roll)", "label": "FilteredRoll"},
                {"var": "deg(madg_euler_pitch)", "label": "FilteredPitch"},
                {"var": "deg(madg_euler_yaw)", "label": "FilteredYaw"},
                {
                    "var": deg(w),
                    "label": "Simu",
                    "color": "black",
                    "linestyle": "--",
                },
            ],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_mahony_pitch(self):
        np.random.seed(134697)

        ctrl = Step(name="ctrl", snames=["u%i" % i for i in range(9)], cons=np.zeros(9))

        angle_ini = -60 * np.pi / 180.0
        wangle = 10.0 * np.pi / 180.0
        sys = TSystem()
        x0 = np.zeros(9)

        # ==================================================
        # Rotation autour de l'axe de tangage
        # ==================================================
        sys.w = np.array([0.0, wangle, 0.0])

        x0[:3] = sys.w
        x0[3:6] = np.array([-np.sin(angle_ini), 0.0, np.cos(angle_ini)])
        x0[6:9] = np.array([0.0, 1.0, 0.0])

        sys.setInitialStateForOutput(x0, "state")

        c = IMU()

        est = MahonyFilter("maho", Kp=0.5, Ki=0.01)
        est.setMagnetometerCalibration(
            offset=np.arange(3), softiron_matrix=np.eye(3) / 2
        )
        b, m = est.getMagnetometerCalibration()
        self.assertAlmostEqual(lin.norm(b - np.arange(3)), 0.0, delta=1.0e-6)
        self.assertAlmostEqual(lin.norm(m - np.eye(3) / 2), 0.0, delta=1.0e-6)
        est.setMagnetometerCalibration(offset=np.zeros(3), softiron_matrix=np.eye(3))

        tfin = -2 * angle_ini / wangle
        tps = np.arange(0.0, tfin, self.dt)
        sim = Simulation()

        sim.addComputer(ctrl)
        sim.addComputer(sys)
        sim.addComputer(c)
        sim.addComputer(est)

        sim.connect("ctrl.setpoint", "sys.command")
        sim.connect("sys.state", "imu.state")
        sim.connect("imu.measurement", "maho.measurement")

        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        # Slower convergence compared to Madgwick
        iok = np.where(tps > 7.0)[0]
        w = angle_ini + tps * wangle
        err_a = np.max(np.abs(self.log.getValue("maho_euler_pitch")[iok] - w[iok]))

        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_a, 0.0, delta=0.06)

        return self.plotVerif(
            "Figure 1",
            [
                {"var": "deg(maho_euler_roll)", "label": "FilteredRoll"},
                {"var": "deg(maho_euler_pitch)", "label": "FilteredPitch"},
                {"var": "deg(maho_euler_yaw)", "label": "FilteredYaw"},
                {
                    "var": deg(w),
                    "label": "Simu",
                    "color": "black",
                    "linestyle": "--",
                },
            ],
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_mahony_yaw(self):
        np.random.seed(134697)

        ctrl = Step(name="ctrl", snames=["u%i" % i for i in range(9)], cons=np.zeros(9))

        angle_ini = -60 * np.pi / 180.0
        wangle = 10.0 * np.pi / 180.0
        sys = TSystem()
        x0 = np.zeros(9)

        # ==================================================
        # Rotation autour de l'axe de lacet
        # ==================================================
        sys.w = np.array([0.0, 0.0, wangle])

        x0[:3] = sys.w
        x0[3:6] = np.array([0.0, 0.0, 1.0])
        x0[6:9] = np.array([np.cos(angle_ini), -np.sin(angle_ini), 0.0])

        sys.setInitialStateForOutput(x0, "state")

        c = IMU()

        est = MahonyFilter("maho", Kp=0.5, Ki=0.01)

        tfin = -2 * angle_ini / wangle
        tps = np.arange(0.0, tfin, self.dt)
        sim = Simulation()

        sim.addComputer(ctrl)
        sim.addComputer(sys)
        sim.addComputer(c)
        sim.addComputer(est)

        sim.connect("ctrl.setpoint", "sys.command")
        sim.connect("sys.state", "imu.state")
        sim.connect("imu.measurement", "maho.measurement")

        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        iok = np.where(tps > 9.0)[0]
        w = angle_ini + tps * wangle
        err_a = np.max(np.abs(self.log.getValue("maho_euler_yaw")[iok] - w[iok]))

        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_a, 0.0, delta=0.08)

        return self.plotVerif(
            "Figure 1",
            [
                {"var": "deg(maho_euler_roll)", "label": "FilteredRoll"},
                {"var": "deg(maho_euler_pitch)", "label": "FilteredPitch"},
                {"var": "deg(maho_euler_yaw)", "label": "FilteredYaw"},
                {
                    "var": deg(w),
                    "label": "Simu",
                    "color": "black",
                    "linestyle": "--",
                },
            ],
        )


if __name__ == "__main__":
    # unittest.main()

    a = TestIMU()
    a.setUp()
    a.test_mahony_yaw()

    plt.show()
