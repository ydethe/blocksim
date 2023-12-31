import numpy as np

from ..utils import FloatArr
from ..control.Controller import AController
from .Quadri import Quadri
from .Motor import Motor


__all__ = ["AttPilot"]


class AttPilot(AController):
    """Inner loop attitude controller

    The input are **setpoint**, **estimation** and **euler**

    **setpoint** has the following parameters:

    * roll (rad)
    * pitch (rad)
    * yaw (rad)
    * A

    **estimation** has the following parameters:

    * px: X coordinate (m)
    * py: Y coordinate (m)
    * pz: Z coordinate (m)
    * vx: X velocity (m/s)
    * vy: Y velocity (m/s)
    * vz: Z velocity (m/s)
    * qw: attitude quaternion real part
    * qx: attitude quaternion x part
    * qy: attitude quaternion y part
    * qz: attitude quaternion z part
    * wx: angular velocity X (rad/s)
    * wy: angular velocity Y (rad/s)
    * wz: angular velocity Z (rad/s)

    **euler** has the following parameters:

    * roll (rad)
    * pitch (rad)
    * yaw (rad)

    The output is **command**

    **command** has the following parameters:

    * s1: velocity setpoint for the motor 1 (rad/s)
    * s2: velocity setpoint for the motor 2 (rad/s)
    * s3: velocity setpoint for the motor 3 (rad/s)
    * s4: velocity setpoint for the motor 4 (rad/s)
    * Gr: TODO
    * Gp: TODO
    * Gy: TODO

    Attributes:
        sys: Quadri instance to be controlled

    Args:
        name: Name of the controller
        sys: Quadri instance to be controlled

    """

    __slots__ = []

    def __init__(self, name: str, sys: Quadri):
        AController.__init__(
            self,
            name,
            shape_setpoint=(4,),
            shape_estimation=(13,),
            snames=["s0", "s1", "s2", "s3", "Gr", "Gp", "Gy"],
        )
        self.defineInput("euler", shape=(3,), dtype=np.float64)
        self.createParameter("sys", sys)

    @property
    def mot(self) -> Motor:
        return self.sys.mot

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: FloatArr,
        estimation: FloatArr,
        command: FloatArr,
        euler: FloatArr,
    ) -> dict:
        (px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz) = estimation
        roll, pitch, yaw = euler
        w = np.array([wx, wy, wz])
        r_cons, p_cons, y_cons, A_cons = setpoint

        J1 = self.sys.J[0, 0]
        J2 = self.sys.J[1, 1]
        J3 = self.sys.J[2, 2]

        a1 = (J2 - J3) / J1
        a2 = (J3 - J1) / J2
        a3 = (J1 - J2) / J3

        # phi / roll / x1
        # theta / pitch / x3
        # psi / yaw / x5
        M = np.array(
            [
                [1.0, np.sin(roll) * np.tan(pitch), np.cos(roll) * np.tan(pitch)],
                [0.0, np.cos(roll), -np.sin(roll)],
                [0.0, np.sin(roll) / np.cos(pitch), np.cos(roll) / np.cos(pitch)],
            ]
        )
        x2, x4, x6 = M @ w

        al1 = 2
        al2 = 2
        al3 = 2
        al4 = 2
        al5 = 8
        al6 = 8

        dx1_cons = 0
        dx3_cons = 0
        dx5_cons = 0
        d2x1_cons = 0
        d2x3_cons = 0
        d2x5_cons = 0

        z1 = roll - r_cons
        z2 = x2 - dx1_cons + al1 * z1
        z3 = pitch - p_cons
        z4 = x4 - dx3_cons + al3 * z3
        z5 = yaw - y_cons
        z6 = x6 - dx5_cons + al5 * z5

        Gr = (-z1 - al2 * z2 - a1 * x4 * x6 + d2x1_cons - al1 * x2 + al1 * dx1_cons) * J1
        Gp = (-z3 - al4 * z4 - a2 * x2 * x6 + d2x3_cons - al3 * x4 + al3 * dx3_cons) * J2
        Gy = (-z5 - al6 * z6 - a3 * x2 * x4 + d2x5_cons - al5 * x6 + al5 * dx5_cons) * J3

        ss1 = (
            A_cons * self.sys.m * self.mot.k * self.sys.l
            + Gy * self.sys.b * self.sys.l
            + 2 * Gp * self.mot.k
        ) / (self.sys.b * self.mot.k * self.sys.l)
        s1 = np.sqrt(max(0, ss1)) / 2

        ss2 = -(
            -A_cons * self.sys.m * self.mot.k * self.sys.l
            + Gy * self.sys.b * self.sys.l
            + 2 * Gr * self.mot.k
        ) / (self.sys.b * self.mot.k * self.sys.l)
        s2 = np.sqrt(max(0, ss2)) / 2

        ss3 = (
            A_cons * self.sys.m * self.mot.k * self.sys.l
            + Gy * self.sys.b * self.sys.l
            - 2 * Gp * self.mot.k
        ) / (self.sys.b * self.mot.k * self.sys.l)
        s3 = np.sqrt(max(0, ss3)) / 2

        ss4 = -(
            -A_cons * self.sys.m * self.mot.k * self.sys.l
            + Gy * self.sys.b * self.sys.l
            - 2 * Gr * self.mot.k
        ) / (self.sys.b * self.mot.k * self.sys.l)
        s4 = np.sqrt(max(0, ss4)) / 2

        u = np.array([s1, s2, s3, s4, Gr, Gp, Gy])

        outputs = {}
        outputs["command"] = u

        return outputs
