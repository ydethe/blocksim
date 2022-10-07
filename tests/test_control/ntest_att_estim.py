import sys
import unittest
from pathlib import Path

import numpy as np
from numpy import pi
import scipy.linalg as lin
from matplotlib import pyplot as plt
import pytest

from blocksim.exceptions import TooWeakAcceleration, TooWeakMagneticField
from blocksim.Simulation import Simulation
from blocksim.core.Generic import GenericComputer
from blocksim.control.System import G6DOFSystem
from blocksim.control.IMU import IMU
from blocksim.control.Controller import PIDController
from blocksim.control.SetPoint import Step
from blocksim.control.Estimator import MadgwickFilter, MahonyFilter
from blocksim.utils import deg, euler_to_quat, geodetic_to_itrf, rad

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestMadgwick(TestBase):
    def setUp(self):
        super().setUp()

        ctrl = Step(name="ctrl", snames=["u%i" % i for i in range(6)], cons=np.zeros(6))

        sys = G6DOFSystem("sys")

        imu = IMU(name="imu")
        cov = np.diag(3 * [np.pi / 180] + 3 * [1e-3 * 9.81] + 3 * [1.0e-6])
        imu.setCovariance(cov)
        moy = np.zeros(9)
        moy[0] = 0.5 * np.pi / 180
        moy[1] = -1.0 * np.pi / 180
        moy[2] = 1.5 * np.pi / 180
        imu.setMean(moy)

        est = MadgwickFilter("madg", beta=2.0)

        sim = Simulation()

        sim.addComputer(ctrl)
        sim.addComputer(sys)
        sim.addComputer(imu)
        sim.addComputer(est)

        sim.connect("ctrl.setpoint", "sys.command")
        sim.connect("sys.state", "imu.state")
        sim.connect("imu.measurement", "madg.measurement")

        self.dt = 5e-2
        self.sys = sys
        self.est = est
        self.sim = sim

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_madgwick_cl(self):
        stp = Step(name="stp", snames=["u0"], cons=np.zeros(1))

        sys = G6DOFSystem("sys")
        x0 = np.zeros(13)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        w = np.array([0.0, 0.0, 0.2])
        x0[6:10] = q
        x0[10:13] = w
        sys.setInitialStateForOutput(x0, output_name="state")

        imu = IMU(name="imu")
        cov = np.diag(3 * [np.pi / 180] + 3 * [1e-3 * 9.81] + 3 * [1.0e-6])
        imu.setCovariance(cov)
        moy = np.zeros(9)
        moy[0] = 0.5 * np.pi / 180
        moy[1] = -1.0 * np.pi / 180
        moy[2] = 1.5 * np.pi / 180
        imu.setMean(moy)

        est = MadgwickFilter("madg", beta=5.0)

        demux = GenericComputer(
            name="demux",
            shape_in=(1,),
            shape_out=(6,),
            callable=lambda x: np.array([0, 0, 0, 0, 0, x[0]]),
            dtype_in=np.float64,
            dtype_out=np.float64,
        )

        ctrl = PIDController(name="ctrl", shape_estimation=2, snames=["u"], coeffs=[1.0, 0.0, 0.0])

        mux = GenericComputer(
            name="mux",
            shape_in=(4,),
            shape_out=(2,),
            callable=lambda x: np.array([x[3], 0.0]),
            dtype_in=np.float64,
            dtype_out=np.float64,
        )

        sim = Simulation()

        sim.addComputer(stp)
        sim.addComputer(sys)
        sim.addComputer(demux)
        sim.addComputer(mux)
        sim.addComputer(ctrl)
        sim.addComputer(imu)
        sim.addComputer(est)

        sim.connect("stp.setpoint", "ctrl.setpoint")
        sim.connect("ctrl.command", "demux.xin")
        sim.connect("demux.xout", "sys.command")
        sim.connect("sys.state", "imu.state")
        sim.connect("imu.measurement", "madg.measurement")
        sim.connect("madg.state", "mux.xin")
        sim.connect("mux.xout", "ctrl.estimation")

        tps = np.arange(200) * self.dt
        sim.simulate(tps, error_on_unconnected=False)
        self.log = sim.getLogger()

        fig = self.plotVerif(
            "ctrl",
            [{"var": "deg(madg_euler_yaw)"}, {"var": "deg(sys_euler_yaw)"}],
            [{"var": "ctrl_command_u"}],
        )

        return fig.render()

    def test_madgwick_exc(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])

        null_acc_meas = np.zeros(9)
        null_acc_meas[-1] = 1
        self.assertRaises(
            TooWeakAcceleration,
            self.est.update,
            0,
            1,
            measurement=null_acc_meas,
            state=q,
            euler=np.zeros(3),
        )

        null_mag_meas = np.zeros(9)
        null_mag_meas[3] = 1
        self.assertRaises(
            TooWeakMagneticField,
            self.est.update,
            0,
            1,
            measurement=null_mag_meas,
            state=q,
            euler=np.zeros(3),
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_madgwick_pitch(self, pb: bool = False):
        angle_ini = -60 * np.pi / 180.0
        # wangle = 10.0 * np.pi / 180.0
        # tfin=60.

        tfin = 60.0
        wangle = -2 * angle_ini / tfin

        # ==================================================
        # Rotation autour de l'axe de tangage
        # ==================================================
        x0 = np.zeros(13)
        x0[:3] = geodetic_to_itrf(lon=rad(1.4433625157254533), lat=rad(43.60441294247197), h=143)
        x0[10:13] = np.array([0.0, wangle, 0.0])
        q = euler_to_quat(roll=0.0, pitch=angle_ini, yaw=pi / 2)
        x0[6:10] = q
        self.sys.setInitialStateForOutput(x0, "state")

        self.est.setMagnetometerCalibration(offset=np.arange(3), softiron_matrix=np.eye(3) / 2)
        b, m = self.est.getMagnetometerCalibration()
        self.assertAlmostEqual(lin.norm(b - np.arange(3)), 0.0, delta=1.0e-6)
        self.assertAlmostEqual(lin.norm(m - np.eye(3) / 2), 0.0, delta=1.0e-6)
        self.est.setMagnetometerCalibration(offset=np.zeros(3), softiron_matrix=np.eye(3))

        tps = np.arange(0.0, tfin, self.dt)

        self.sim.simulate(tps, progress_bar=pb)

        self.log = self.sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        iok = np.where(tps > 2.0)[0]
        w = angle_ini + tps * wangle
        err_a = np.max(np.abs(self.log.getValue("madg_euler_pitch")[iok] - w[iok]))

        # self.assertAlmostEqual(err_t, 0.0, delta=1e-9)
        # self.assertAlmostEqual(err_a, 0.0, delta=5e-2)

        fig = self.plotVerif(
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
        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_madgwick_yaw(self):
        angle_ini = -60 * np.pi / 180.0
        wangle = 10.0 * np.pi / 180.0

        # ==================================================
        # Rotation autour de l'axe de lacet
        # ==================================================
        x0 = np.zeros(13)
        x0[10:13] = np.array([0.0, 0.0, wangle])
        q = euler_to_quat(roll=0.0, pitch=0, yaw=angle_ini)
        x0[6:10] = q
        self.sys.setInitialStateForOutput(x0, "state")

        tfin = -2 * angle_ini / wangle
        tps = np.arange(0.0, tfin, self.dt)

        self.sim.simulate(tps, progress_bar=False)

        self.log = self.sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        iok = np.where(tps > 2.0)[0]
        w = angle_ini + tps * wangle
        err_a = np.max(np.abs(self.log.getValue("madg_euler_yaw")[iok] - w[iok]))

        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-9)
        self.assertAlmostEqual(err_a, 0.0, delta=5e-2)

        fig = self.plotVerif(
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
        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_madgwick_all_dof(self):
        angle_ini = -60 * np.pi / 180.0
        wangle = 10.0 * np.pi / 180.0

        # ==================================================
        # Rotation autour de l'axe de tangage
        # ==================================================
        x0 = np.zeros(13)
        w = np.random.rand(3) * 2 - 1
        x0[10:13] = w
        q = np.random.rand(4) * 2 - 1
        x0[6:10] = q / lin.norm(q)
        self.sys.setInitialStateForOutput(x0, "state")
        A = np.random.randint(low=-5, high=5, size=(3, 3))
        self.sys.J = A @ A.T

        tfin = 5.0
        tps = np.arange(0.0, tfin, self.dt)

        self.sim.simulate(tps, progress_bar=False)

        self.log = self.sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        iok = np.where(tps > 0.5)[0]
        mp = self.log.getValue("deg(madg_euler_pitch)")
        sp = self.log.getValue("deg(sys_euler_pitch)")
        err_a = np.max(np.abs(sp - mp)[iok])

        self.assertAlmostEqual(err_a, 0.0, delta=6.0)

        fig = self.plotVerif(
            "Figure 1",
            [
                {"title": "Pitch (deg)", "coord": 0},
                {"var": "deg(madg_euler_pitch)"},
                {
                    "var": "deg(sys_euler_pitch)",
                    "label": "Simu",
                    "color": "black",
                    "linestyle": "--",
                },
            ],
        )
        return fig.render()


class TestMahony(TestBase):
    def setUp(self):
        super().setUp()

        ctrl = Step(name="ctrl", snames=["u%i" % i for i in range(6)], cons=np.zeros(6))

        sys = G6DOFSystem("sys")

        imu = IMU(name="imu")
        cov = np.diag(3 * [np.pi / 180] + 3 * [1e-3 * 9.81] + 3 * [1.0e-6])
        imu.setCovariance(cov)
        moy = np.zeros(9)
        moy[0] = 0.5 * np.pi / 180
        moy[1] = -1.0 * np.pi / 180
        moy[2] = 1.5 * np.pi / 180
        imu.setMean(moy)

        est = MahonyFilter("maho", Kp=0.5, Ki=0.01)

        sim = Simulation()

        sim.addComputer(ctrl)
        sim.addComputer(sys)
        sim.addComputer(imu)
        sim.addComputer(est)

        sim.connect("ctrl.setpoint", "sys.command")
        sim.connect("sys.state", "imu.state")
        sim.connect("imu.measurement", "maho.measurement")

        self.dt = 1e-2
        self.sys = sys
        self.est = est
        self.sim = sim

    def test_mahony_exc(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])

        null_acc_meas = np.zeros(9)
        null_acc_meas[-1] = 1
        self.assertRaises(
            TooWeakAcceleration,
            self.est.update,
            0,
            1,
            measurement=null_acc_meas,
            state=q,
            euler=np.zeros(3),
        )

        null_mag_meas = np.zeros(9)
        null_mag_meas[3] = 1
        self.assertRaises(
            TooWeakMagneticField,
            self.est.update,
            0,
            1,
            measurement=null_mag_meas,
            state=q,
            euler=np.zeros(3),
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_mahony_pitch(self):
        angle_ini = -60 * np.pi / 180.0
        wangle = 10.0 * np.pi / 180.0

        # ==================================================
        # Rotation autour de l'axe de tangage
        # ==================================================
        x0 = np.zeros(13)
        x0[10:13] = np.array([0.0, wangle, 0.0])
        q = euler_to_quat(roll=0.0, pitch=angle_ini, yaw=pi / 2)
        x0[6:10] = q
        self.sys.setInitialStateForOutput(x0, "state")

        self.est.Ki = 0.0
        self.est.setMagnetometerCalibration(offset=np.arange(3), softiron_matrix=np.eye(3) / 2)
        b, m = self.est.getMagnetometerCalibration()
        self.assertAlmostEqual(lin.norm(b - np.arange(3)), 0.0, delta=1.0e-6)
        self.assertAlmostEqual(lin.norm(m - np.eye(3) / 2), 0.0, delta=1.0e-6)
        self.est.setMagnetometerCalibration(offset=np.zeros(3), softiron_matrix=np.eye(3))

        tfin = -2 * angle_ini / wangle
        tps = np.arange(0.0, tfin, self.dt)

        self.sim.simulate(tps, progress_bar=False)

        self.log = self.sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        # Slower convergence compared to Madgwick
        iok = np.where(tps > 7.0)[0]
        w = angle_ini + tps * wangle
        err_a = np.max(np.abs(self.log.getValue("maho_euler_pitch")[iok] - w[iok]))

        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-9)
        self.assertAlmostEqual(err_a, 0.0, delta=0.085)

        fig = self.plotVerif(
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
        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_mahony_yaw(self):
        angle_ini = -60 * np.pi / 180.0
        wangle = 10.0 * np.pi / 180.0

        # ==================================================
        # Rotation autour de l'axe de lacet
        # ==================================================
        x0 = np.zeros(13)
        x0[10:13] = np.array([0.0, 0.0, wangle])
        q = euler_to_quat(roll=0.0, pitch=0, yaw=angle_ini)
        x0[6:10] = q
        self.sys.setInitialStateForOutput(x0, "state")

        tfin = -2 * angle_ini / wangle
        tps = np.arange(0.0, tfin, self.dt)

        self.sim.simulate(tps, progress_bar=False)

        self.log = self.sim.getLogger()

        err_t = np.max(np.abs(self.log.getValue("t") - tps))

        iok = np.where(tps > 9.0)[0]
        w = angle_ini + tps * wangle
        err_a = np.max(np.abs(self.log.getValue("maho_euler_yaw")[iok] - w[iok]))

        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_a, 0.0, delta=0.08)

        fig = self.plotVerif(
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
        return fig.render()


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestMadgwick()
    a.setUp()
    # a.test_madgwick_all_dof()
    a.test_madgwick_pitch(pb=True)

    showFigures()
