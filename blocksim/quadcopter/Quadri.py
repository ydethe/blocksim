from typing import Iterable

import numpy as np
import scipy.linalg as lin

from blocksim.core.Node import Input
from blocksim.control.System import G6DOFSystem
from blocksim.utils import vecBodyToEarth, vecEarthToBody


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
        )
        inp = Input("command", shape=(4,), dtype=np.float64)
        self.replaceInput(old_name="command", new_input=inp)
        self.m = 0.458
        self.J = np.diag([14.6, 10.8, 7.8]) * 1e-3
        self.createParameter("mot", mot)
        self.createParameter("b", 3.8e-6)
        self.createParameter("g", 9.81)
        self.createParameter("l", 22.5e-2)
        self.createParameter("Jr", 3.4e-5)

    def getActions(self, x: np.array, u: np.array) -> Iterable[np.array]:
        s1, s2, s3, s4 = u
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = x
        att = np.array([qw, qx, qy, qz])

        e3 = np.array([0, 0, 1])
        t3 = vecBodyToEarth(att, e3)
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
        force, torque = self.getActions(x, u)

        a = np.hstack((force, torque))

        return G6DOFSystem.transition(self, t, x, a)

    def getEquilibriumSpeed(self):
        s_eq = np.sqrt(self.m * self.g / 4 / self.b)
        return s_eq
