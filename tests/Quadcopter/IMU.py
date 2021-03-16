import numpy as np

from blocksim.blocks.Sensors import ASensors
from blocksim.utils import quat_to_matrix, quat_to_euler


class IMU(ASensors):
    __slots__ = []

    def __init__(self):
        ASensors.__init__(
            self,
            "imu",
            name_of_outputs=["gx", "gy", "gz", "ax", "ay", "az", "mx", "my", "mz"],
            name_of_inputs=["state"],
        )

    def compute_state(self, t1: float, t2: float, inputs: dict) -> np.array:
        """Updates the state of the element

        Called at each simulation step

        Args:
          t1
            Current date (s)
          t2
            The date after the update (s)
          inputs
            Dictionnary of the inputs :

            * key = name of the source element
            * value = source's state vector

        Returns:
          The new state of the element

        """
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = self.getDataForInput(
            inputs, "state"
        )
        R = quat_to_matrix(qw, qx, qy, qz)
        gx = wx
        gy = wy
        gz = wz
        ax, ay, az = R @ np.array([0, 0, 1]) * 9.81
        mx, my, mz = R @ np.array([1, 0, 0])
        return np.array([gx, gy, gz, ax, ay, az, mx, my, mz])
