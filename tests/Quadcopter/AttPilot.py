import numpy as np
import scipy.linalg as lin

from SystemControl.blocks.Controller import AController


# name_of_outputs=['s0_cons','s1_cons','s2_cons','s3_cons']
class AttPilot(AController):
    __slots__ = []

    def __init__(self, sys, mot):
        name_of_outputs = ["s0_cons", "s1_cons", "s2_cons", "s3_cons"]
        name_of_states = ["state_" + x for x in name_of_outputs]
        name_of_states.extend(["Gr", "Gp", "Gy"])
        AController.__init__(
            self,
            "ctlatt",
            name_of_outputs=name_of_outputs,
            name_of_states=name_of_states,
        )
        self.createParameter("sys", sys)
        self.createParameter("mot", mot)

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
        (
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
        ) = self.getDataForInput(inputs, "estimation")
        w = np.array([wx, wy, wz])
        r_cons, p_cons, y_cons, A_cons = self.getDataForInput(inputs, "setpoint")

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

        Gr = (
            -z1 - al2 * z2 - a1 * x4 * x6 + d2x1_cons - al1 * x2 + al1 * dx1_cons
        ) * J1
        Gp = (
            -z3 - al4 * z4 - a2 * x2 * x6 + d2x3_cons - al3 * x4 + al3 * dx3_cons
        ) * J2
        Gy = (
            -z5 - al6 * z6 - a3 * x2 * x4 + d2x5_cons - al5 * x6 + al5 * dx5_cons
        ) * J3

        # P = self.sys.m*self.sys.g
        s1 = (
            np.sqrt(
                (
                    A_cons * self.sys.m * self.mot.k * self.sys.l
                    + Gy * self.sys.b * self.sys.l
                    + 2 * Gp * self.mot.k
                )
                / (self.sys.b * self.mot.k * self.sys.l)
            )
            / 2
        )
        s2 = (
            np.sqrt(
                -(
                    -A_cons * self.sys.m * self.mot.k * self.sys.l
                    + Gy * self.sys.b * self.sys.l
                    + 2 * Gr * self.mot.k
                )
                / (self.sys.b * self.mot.k * self.sys.l)
            )
            / 2
        )
        s3 = (
            np.sqrt(
                (
                    A_cons * self.sys.m * self.mot.k * self.sys.l
                    + Gy * self.sys.b * self.sys.l
                    - 2 * Gp * self.mot.k
                )
                / (self.sys.b * self.mot.k * self.sys.l)
            )
            / 2
        )
        s4 = (
            np.sqrt(
                -(
                    -A_cons * self.sys.m * self.mot.k * self.sys.l
                    + Gy * self.sys.b * self.sys.l
                    - 2 * Gr * self.mot.k
                )
                / (self.sys.b * self.mot.k * self.sys.l)
            )
            / 2
        )

        return np.array([s1, s2, s3, s4, Gr, Gp, Gy])

    def compute_output(self, t: float, state: np.array, inputs: dict) -> np.array:
        return state[:4]
