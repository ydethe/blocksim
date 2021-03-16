import numpy as np
import scipy.linalg as lin

from SystemControl.blocks.System import ASystem


# name_of_outputs=['s%i' % num,'ds%i' % num]
class Motor(ASystem):
    """"""

    __slots__ = []

    def __init__(self, num):
        ASystem.__init__(
            self,
            "mot%i" % num,
            name_of_states=["state_s%i" % num],
            name_of_outputs=["s%i" % num, "ds%i" % num],
        )
        self.createParameter("num", num)
        self.createParameter("km", 4.3e-3)
        self.createParameter("Jr", 3.4e-5)
        self.createParameter("R", 0.67)
        self.createParameter("kgb", 2.7e-3)
        self.createParameter("k", 2.9e-5)
        self.createParameter("Umax", 12)

    def transition(self, t: float, x: np.array, u: np.array) -> np.array:
        (s,) = x
        (u0,) = u
        ds = (
            -self.km ** 2 / (self.Jr * self.R) * s
            - self.kgb * self.k / self.Jr * np.abs(s) * s
            + self.km / (self.Jr * self.R) * np.clip(u0, -self.Umax, self.Umax)
        )

        dX = np.array([ds])
        return dX

    def compute_output(self, t: float, state: np.array, inputs: dict) -> np.array:
        (s,) = state
        u = self.getDataForInput(inputs, "command")
        (ds,) = self.transition(t, state, u)
        return np.array([s, ds])
