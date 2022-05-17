from typing import Any

from nptyping import NDArray, Shape
import numpy as np

from ..control.System import ASystem


# name_of_outputs=['s%i' % num,'ds%i' % num]
class Motor(ASystem):
    """Motor with its propeller

    Attributes:
        num: the 0-based index of the motor
        km: motor constant (N.m/A)
        Jr: rotor inertia (J.g.m²)
        R: motor resistance (ohm)
        kgb: gearbox ratio (-)
        k: drag coefficient
        Umax: maximal voltage (V)

    Args:
        prefix: name prefix of the Motor
        num: Index of the motor (starting at 0)

    """

    __slots__ = []

    def __init__(self, prefix: str, num: int):
        ASystem.__init__(
            self,
            f"{prefix}{num}",
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

    def transition(
        self, t: float, x: NDArray[Any, Any], u: NDArray[Any, Any]
    ) -> NDArray[Any, Any]:
        (s,) = x
        (u0,) = u
        ds = (
            -self.km**2 / (self.Jr * self.R) * s
            - self.kgb * self.k / self.Jr * np.abs(s) * s
            + self.km / (self.Jr * self.R) * np.clip(u0, -self.Umax, self.Umax)
        )

        dX = np.array([ds])
        return dX

    def update(
        self,
        t1: float,
        t2: float,
        command: NDArray[Any, Any],
        state: NDArray[Any, Any],
        vel: NDArray[Any, Any],
    ) -> dict:
        outputs = super().update(t1, t2, command, state)

        (s,) = outputs["state"]
        (ds,) = self.transition(t2, (s,), command)

        outputs["vel"] = np.array([s, ds])

        return outputs
