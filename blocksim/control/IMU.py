import numpy as np

from ..control.Sensors import ASensors
from ..utils import FloatArr, quat_to_matrix
from .Earth6DDLPosition import Earth6DDLPosition


class IMU(ASensors):
    """Implementation of noisy IMU model

    Args:
        name: Name of the IMU

    """

    __slots__ = []

    def __init__(self, name: str):
        ASensors.__init__(
            self,
            name=name,
            shape_state=13,
            snames=["gx", "gy", "gz", "ax", "ay", "az", "mx", "my", "mz"],
        )

    def update(
        self,
        t1: float,
        t2: float,
        state: FloatArr,
        measurement: FloatArr,
    ) -> dict:
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = state

        ep = Earth6DDLPosition(
            name="",
            time=None,
            position=np.array([px, py, pz]),
            velocity=np.array([vx, vy, vz]),
            attitude=np.array([qw, qx, qy, qz]),
            angular_rate=np.array([wx, wy, wz]),
        )

        R = quat_to_matrix(qw, qx, qy, qz)
        gx = wx
        gy = wy
        gz = wz
        ax, ay, az = R.T @ np.array([0, 0, 1]) * 9.81
        mx, my, mz = R.T @ ep.magneticDeclination(frame="NED")
        meas = np.array([gx, gy, gz, ax, ay, az, mx, my, mz])

        outputs = {}
        outputs["measurement"] = meas

        return outputs
