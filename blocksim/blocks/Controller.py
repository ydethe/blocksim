import numpy as np

from ..core.Frame import Frame
from ..core.Node import AComputer

__all__ = ["AController", "PIDController"]


class AController(AComputer):
    """Abstract class for a scalar controller

    Implement the method **updateAllOutput** to make it concrete

    The inputs of the computer are **estimation** and **setpoint**
    The output of the computer is **command**

    Args:
      name
        Name of the element
      nscal_estimation
        Number of scalars in the data expected by the estimation

    """

    def __init__(self, name: str, nscal_estimation: int):
        AComputer.__init__(self, name)
        self.defineInput("setpoint", nscal=1, dtype=np.float64)
        self.defineInput("estimation", nscal=nscal_estimation, dtype=np.float64)
        self.defineOutput("command", nscal=1, dtype=np.float64)
        self.setInitialStateForOutput(np.array([0]), "command")


class PIDController(AController):
    """One-dimensional PID controller

    The inputs of the computer are **estimation** and **setpoint**
    The outputs of the computer are **command** and **integral**

    The **estimation** is an estimation of the system, with the following constraints:
    * the first value of **estimation** is the position
    * the second value of **estimation** is the velocity

    Args:
      name
        Name of the element
      nscal_estimation
        Number of scalars in the data expected by the estimation (> 2)
      coeffs
        Coefficients of the retroaction (P, I, D)

    """

    def __init__(self, name: str, nscal_estimation: int, coeffs: float):
        AController.__init__(self, name, nscal_estimation)
        self.defineOutput("integral", nscal=1, dtype=np.float64)
        self.setInitialStateForOutput(np.array([0]), "integral")
        self.__coeffs = coeffs

    def updateAllOutput(self, frame: Frame):
        stp = self.getInputByName("setpoint")
        state = self.getInputByName("estimation")

        cmd = self.getOutputByName("command")
        itg = self.getOutputByName("integral")

        (c,) = stp.getDataForFrame(frame)
        data = state.getDataForFrame(frame)
        x = data[0]
        dx = data[1]
        (ix,) = itg.getDataForFrame(frame)

        P, I, D = self.__coeffs
        u = P * (x - c) + I * ix + D * dx

        dt = frame.getTimeStep()
        itg.setData(np.array([ix + dt * (x - c)]))
        cmd.setData(-np.array([u]))
