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

    """

    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("setpoint")
        self.defineInput("estimation")
        self.defineOutput("command")
        self.setInitialStateForOutput(np.array([0]), "command")


class PController(AController):
    """Proportionnal scalar controller

    Implement the method **compute_state** to make it concrete

    The inputs of the computer are **estimation** and **setpoint**
    The output of the computer is **command**

    Args:
      name
        Name of the element
      coeff_P
        Coefficient of the proportionnal retroaction

    """

    def __init__(self, name: str, coeff_P: float):
        AController.__init__(self, name)
        self.__coeff_P = coeff_P

    def updateAllOutput(self, frame: Frame):
        stp = self.getDataForInput(frame, name="setpoint")
        X = self.getDataForInput(frame, name="estimation")

        otp = self.getOutputByName("command")

        u = -self.__coeff_P * (X - stp)

        otp.setData(u)
