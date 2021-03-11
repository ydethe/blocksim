import numpy as np

from ..core.Frame import Frame
from ..core.Node import AComputer

__all__ = ["AController", "PController"]


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


class PController(AController):
    """Proportionnal scalar controller

    Implement the method **compute_state** to make it concrete

    The inputs of the computer are **estimation** and **setpoint**
    The output of the computer is **command**

    Args:
      name
        Name of the element
      nscal_estimation
        Number of scalars in the data expected by the estimation
      coeff_P
        Coefficient of the proportionnal retroaction

    """

    def __init__(self, name: str, nscal_estimation: int, coeff_P: float):
        AController.__init__(self, name, nscal_estimation)
        self.__coeff_P = coeff_P

    def updateAllOutput(self, frame: Frame):
        stp = self.getDataForInput(frame, name="setpoint")
        X = self.getDataForInput(frame, name="estimation")

        otp = self.getOutputByName("command")

        u = -self.__coeff_P * np.array([X[0] - stp[0]])

        otp.setData(u)
