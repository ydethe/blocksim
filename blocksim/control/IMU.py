import numpy as np

from ..control.Sensors import ASensors
from ..utils import quat_to_matrix, quat_to_euler


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

    def compute_outputs(
        self, t1: float, t2: float, state: np.array, measurement: np.array
    ) -> dict:
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = state

        R = quat_to_matrix(qw, qx, qy, qz)
        gx = wx
        gy = wy
        gz = wz
        ax, ay, az = R.T @ np.array([0, 0, 1]) * 9.81
        mx, my, mz = R.T @ np.array([1, 0, 0])
        meas = np.array([gx, gy, gz, ax, ay, az, mx, my, mz])

        outputs = {}
        outputs["measurement"] = meas

        return outputs
