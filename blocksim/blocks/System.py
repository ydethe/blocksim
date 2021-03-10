from abc import abstractmethod

import numpy as np
from scipy.integrate import ode

from ..core.Frame import Frame
from ..core.Node import AComputer

__all__ = ["ASystem", "PController"]


class ASystem(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("command")
        self.defineOutput("output")

        self.__integ = ode(self.transition).set_integrator("zvode", method="bdf")

    @abstractmethod
    def transition(self, t: float, y: np.array, u: np.array) -> np.array:
        pass

    def updateAllOutput(self, frame: Frame):
        u = self.getDataFromInput(frame, name="command")
        y0 = self.getDataForOuput(frame, name="output")

        t0 = frame.getStartTimeStamp()
        t1 = frame.getStopTimeStamp()

        self.__integ.set_initial_value(y0, t0).set_f_params(u).set_jac_params(u)
        y1 = self.__integ.integrate(t1)

        otp = self.getOutputByName("output")
        otp.setData(y1)
