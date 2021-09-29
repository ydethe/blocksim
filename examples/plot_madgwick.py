r"""
Madgwick attitude estimator
===========================

"""
###############################################################################
# Main libraries import
# ---------------------

import numpy as np

from blocksim.Simulation import Simulation
from blocksim.control.System import ASystem
from blocksim.control.Sensors import ASensors
from blocksim.control.SetPoint import Step
from blocksim.control.Estimator import MadgwickFilter
from blocksim.utils import deg


###############################################################################
# Definition of the 6DDF system
# -----------------------------
# We define a simple system that just rotates with an angular velocity given by its parameter w
# The output of this system are the (noiseless) state of gyrometer (x3), accelerometer (x3) and magnetometer (x3)


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


###############################################################################
# Definition of a noisy IMU
# -------------------------
# We define a simple IMU that just copies the states of gyrometer (x3), accelerometer (x3) and magnetometer (x3),
# and adds a gaussian noise and a bias


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


###############################################################################
# Construction of the simulation
# ------------------------------

sim = Simulation()

###############################################################################
# Definition of a null Step function

ctrl = Step(name="ctrl", snames=["u%i" % i for i in range(9)], cons=np.zeros(9))

sim.addComputer(ctrl)

###############################################################################
# Initialisation of the TSystem, rotating about the pitch axis

angle_ini = -60 * np.pi / 180.0
wangle = 10.0 * np.pi / 180.0
sys = TSystem()

sys.w = np.array([0.0, wangle, 0.0])

x0 = np.zeros(9)
x0[:3] = sys.w
x0[3:6] = np.array([-np.sin(angle_ini), 0.0, np.cos(angle_ini)])
x0[6:9] = np.array([0.0, 1.0, 0.0])

sys.setInitialStateForOutput(x0, "state")

sim.addComputer(sys)

###############################################################################
# Initialisation of the IMU

c = IMU()

sim.addComputer(c)

###############################################################################
# Initialisation of the Madgwick attitude estimator

est = MadgwickFilter("madg", beta=2.0)
est.setMagnetometerCalibration(offset=np.arange(3), softiron_matrix=np.eye(3) / 2)
b, m = est.getMagnetometerCalibration()
est.setMagnetometerCalibration(offset=np.zeros(3), softiron_matrix=np.eye(3))

sim.addComputer(est)

###############################################################################
# Simulation
# ----------

from blocksim.Graphics import plotVerif
from matplotlib import pyplot as plt

sim.connect("ctrl.setpoint", "sys.command")
sim.connect("sys.state", "imu.state")
sim.connect("imu.measurement", "madg.measurement")

tfin = -2 * angle_ini / wangle
tps = np.arange(0.0, tfin, 1e-2)
w = angle_ini + tps * wangle

sim.simulate(tps, progress_bar=False)
log = sim.getLogger()

plotVerif(
    log,
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

plt.show()
