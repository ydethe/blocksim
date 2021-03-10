import numpy as np

from ..core.Frame import Frame
from ..core.Node import AComputer

__all__ = ["AController", "PController"]


class AController(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("setpoint")
        self.defineInput("estimation")
        self.defineOutput("command")
        self.setInitialStateForOutput(np.array([0]), "command")


class PController(AController):
    def __init__(self, name: str, coeff_P: float):
        AController.__init__(self, name)
        self.__coeff_P = coeff_P

    def updateAllOutput(self, frame: Frame):
        stp = self.getDataFromInput(frame, name="setpoint")
        yest, vest = self.getDataFromInput(frame, name="estimation")

        otp = self.getOutputByName("command")

        u = -self.__coeff_P * (yest - stp)

        otp.setData(u)
