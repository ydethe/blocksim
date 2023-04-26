from typing import Any

from nptyping import NDArray
import numpy as np
from numpy import pi

from ahrs.filters import AngularRate
from ahrs.filters import Madgwick
from ahrs.filters import Tilt
from ahrs.filters.aqua import AQUA
from ahrs.filters.ekf import EKF

from blocksim.core.Node import AComputer
from blocksim.Simulation import Simulation
from blocksim.control.System import G6DOFSystem
from blocksim.control.IMU import IMU
from blocksim.control.SetPoint import Step
from blocksim.utils import deg, euler_to_quat, geodetic_to_itrf, quat_to_euler, rad
from blocksim.graphics import plotVerif, showFigures


class AHRSFilter(AComputer):

    __slots__ = ["__ahrs"]

    def __init__(self, name: str, algo: str):
        AComputer.__init__(self, name)

        if algo == "AngularRate":
            self.__ahrs = AngularRate()
        elif algo == "Madgwick":
            self.__ahrs = Madgwick(gain=2.0)
        elif algo == "EKF":
            self.__ahrs = EKF()
        elif algo == "AQUA":
            self.__ahrs = AQUA()

        self.defineInput("measurement", shape=(9,), dtype=np.float64)
        self.defineOutput("state", snames=["q0", "q1", "q2", "q3"], dtype=np.float64)
        self.defineOutput("euler", snames=["roll", "pitch", "yaw"], dtype=np.float64)
        self.setInitialStateForOutput(np.array([1, 0, 0, 0]), "state")

        self.createParameter(name="algo", value=algo)

    def update(
        self,
        t1: float,
        t2: float,
        measurement: NDArray[Any, Any],
        state: NDArray[Any, Any],
        euler: NDArray[Any, Any],
    ) -> dict:
        dt = t2 - t1

        gyr = measurement[0:3]
        acc = measurement[3:6]
        mag = measurement[6:9]

        if dt == 0:
            tilt = Tilt()
            state = tilt.estimate(acc=acc, mag=mag * 1e3)

        # https://ahrs.readthedocs.io/en/latest/filters/madgwick.html#ahrs.filters.madgwick.Madgwick.updateMARG
        if self.algo == "AngularRate":
            self.__ahrs.Dt = dt
            new_q = self.__ahrs.update(q=state, gyr=gyr, method="closed")
        elif self.algo == "Madgwick":
            self.__ahrs.Dt = dt
            new_q = self.__ahrs.updateMARG(q=state, gyr=gyr, acc=acc, mag=mag * 1e9)
        elif self.algo == "AQUA":
            self.__ahrs.Dt = dt
            new_q = self.__ahrs.updateMARG(q=state, gyr=gyr, acc=acc, mag=mag * 1e3)
        elif self.algo == "EKF":
            self.__ahrs.Dt = dt
            new_q = self.__ahrs.update(q=state, gyr=gyr, acc=acc, mag=mag * 1e6)
        elif self.algo == "TILT":
            tilt = Tilt()
            new_q = tilt.estimate(acc=acc, mag=mag * 1e3)

        output = {}
        output["state"] = new_q
        output["euler"] = np.array(quat_to_euler(*new_q))

        # q = Quaternion(new_q)
        # print(q.to_angles())
        # print(output["euler"])
        # print(mag)
        # print(72*'-')

        return output


class TestAHRS(object):
    def setUp(self):
        ctrl = Step(name="ctrl", snames=["u%i" % i for i in range(6)], cons=np.zeros(6))

        sys = G6DOFSystem("sys")

        imu = IMU(name="imu")
        cov = np.diag(3 * [np.pi / 180] + 3 * [1e-3 * 9.81] + 3 * [(100e-9) ** 2])
        imu.setCovariance(cov)
        moy = np.zeros(9)
        moy[0] = 0.5 * np.pi / 180
        moy[1] = -1.0 * np.pi / 180
        moy[2] = 1.5 * np.pi / 180
        imu.setMean(moy)

        # est = AHRSFilter("ahrs", algo="AngularRate")
        # est = AHRSFilter("ahrs", algo="Madgwick")
        # est = AHRSFilter("ahrs", algo="AQUA")
        est = AHRSFilter("ahrs", algo="TILT")
        # est = AHRSFilter("ahrs", algo="EKF")

        sim = Simulation()

        sim.addComputer(ctrl)
        sim.addComputer(sys)
        sim.addComputer(imu)
        sim.addComputer(est)

        sim.connect("ctrl.setpoint", "sys.command")
        sim.connect("sys.state", "imu.state")
        sim.connect("imu.measurement", "ahrs.measurement")

        self.dt = 5e-2
        self.sys = sys
        self.est = est
        self.sim = sim

    def test_poney(self, pb=False):
        angle_ini = -45 * np.pi / 180.0
        # wangle = 10.0 * np.pi / 180.0
        # tfin=60.

        tfin = 30
        wangle = -2 * angle_ini / tfin

        # ==================================================
        # Rotation autour de l'axe de tangage
        # ==================================================
        x0 = np.zeros(13)
        x0[:3] = geodetic_to_itrf(lon=rad(1.4433625157254533), lat=rad(43.60441294247197), h=143)
        x0[10:13] = np.array([0.0, wangle, 0.0])
        q = euler_to_quat(roll=0.0, pitch=angle_ini, yaw=pi / 6)
        x0[6:10] = q
        self.sys.setInitialStateForOutput(x0, "state")

        tps = np.arange(0.0, tfin, self.dt)

        self.sim.simulate(tps, progress_bar=pb)

        self.log = self.sim.getLogger()

        w = angle_ini + tps * wangle

        fig = plotVerif(
            self.log,
            "Figure 1",
            [
                {"var": "imu_measurement_mx", "label": "Mx", "color": "red"},
                {"var": "imu_measurement_my", "label": "My", "color": "green"},
                {"var": "imu_measurement_mz", "label": "Mz", "color": "blue"},
            ],
            [
                {"var": "imu_measurement_ax", "label": "Ax", "color": "red"},
                {"var": "imu_measurement_ay", "label": "Ay", "color": "green"},
                {"var": "imu_measurement_az", "label": "Az", "color": "blue"},
            ],
        )

        fig = plotVerif(
            self.log,
            "Figure 1",
            [
                {"var": "deg(ahrs_euler_roll)", "label": "FilteredRoll"},
                {"var": "deg(ahrs_euler_pitch)", "label": "FilteredPitch"},
                {"var": "deg(ahrs_euler_yaw)", "label": "FilteredYaw"},
                {
                    "var": deg(w),
                    "label": "Simu",
                    "color": "black",
                    "linestyle": "--",
                },
                {"var": 30 + 0 * w, "color": "black", "linestyle": "--"},
                {"var": 0 * w, "color": "black", "linestyle": "--"},
            ],
        )

        return fig.render()


if __name__ == "__main__":
    a = TestAHRS()
    a.setUp()
    a.test_poney(pb=True)

    showFigures()
