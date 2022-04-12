import numpy as np
import scipy.linalg as lin

from blocksim.control.System import ASystem


# name_of_outputs=['s%i' % num,'ds%i' % num]
class Motor(ASystem):
    """"""

    __slots__ = []

    def __init__(self, num):
        ASystem.__init__(
            self,
            "mot%i" % num,
            shape_command=(1,),
            snames_state=["s"],
            dtype=np.float64,
        )
        self.defineOutput("vel", snames=["s", "ds"], dtype=np.float64)
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

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        command: np.array,
        state: np.array,
        vel: np.array,
    ) -> dict:
        outputs = super().compute_outputs(t1, t2, command, state)

        (s,) = outputs["state"]
        (ds,) = self.transition(t2, (s,), command)

        outputs["vel"] = np.array([s, ds])

        return outputs
