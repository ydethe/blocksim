from typing import Iterable

import numpy as np
import scipy.linalg as lin

from SystemControl.blocks.System import G6DOFSystem
from SystemControl.utils import quat_to_matrix, quat_to_euler


# name_of_outputs=['px','py','pz','vx','vy','vz','roll','pitch','yaw','wx','wy','wz']
class Quadri(G6DOFSystem):
    """

    http://www.gipsa-lab.grenoble-inp.fr/~nicolas.marchand/teaching/Nonlinear_PSPI.pdf

    """

    __slots__ = []

    def __init__(self, mot):
        G6DOFSystem.__init__(
            self,
            "sys",
            name_of_outputs=[
                "px",
                "py",
                "pz",
                "vx",
                "vy",
                "vz",
                "fx",
                "fy",
                "fz",
                "roll",
                "pitch",
                "yaw",
                "wx",
                "wy",
                "wz",
                "tx",
                "ty",
                "tz",
            ],
        )
        self.m = 0.458
        self.J = np.diag([14.6, 10.8, 7.8]) * 1e-3
        self.createParameter("mot", mot)
        self.createParameter("b", 3.8e-6)
        self.createParameter("g", 9.81)
        self.createParameter("l", 22.5e-2)
        self.createParameter("Jr", 3.4e-5)

    def getActions(self, u: np.array) -> Iterable[np.array]:
        s1, s2, s3, s4 = u

        e3 = np.array([0, 0, 1])
        t3 = self.vecBodyToEarth(e3)
        force = -self.m * self.g * e3 + self.b * np.sum(u ** 2) * t3
        torque = np.array(
            [
                (s4 ** 2 - s2 ** 2) * self.l * self.b,
                (s1 ** 2 - s3 ** 2) * self.l * self.b,
                (s1 ** 2 + s3 ** 2 - s2 ** 2 - s4 ** 2) * self.mot.k,
            ]
        )

        return force, torque

    def transition(self, t: float, x: np.array, u: np.array) -> np.array:
        force, torque = self.getActions(u)

        a = np.hstack((force, torque))

        return G6DOFSystem.transition(self, t, x, a)

    def getEquilibriumSpeed(self):
        s_eq = np.sqrt(self.m * self.g / 4 / self.b)
        return s_eq

    def compute_output(self, t: float, state: np.array, inputs: dict) -> np.array:
        """Function that computes the output of the element from its state and its inputs.

        Args:
          t
            Date of the current state (s)
          state
            Current state
          inputs
            Dictionnary of the inputs :

            * key = name of the source element
            * value = source's state vector

        """
        u = self.getDataForInput(inputs, "command")
        force, torque = self.getActions(u)
        fx, fy, fz = force
        tx, ty, tz = torque
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = state
        roll, pitch, yaw = quat_to_euler(qw, qx, qy, qz)
        y = np.array(
            [
                px,
                py,
                pz,
                vx,
                vy,
                vz,
                fx,
                fy,
                fz,
                roll,
                pitch,
                yaw,
                wx,
                wy,
                wz,
                tx,
                ty,
                tz,
            ]
        )
        return y
