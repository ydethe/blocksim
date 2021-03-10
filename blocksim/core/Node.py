from typing import Iterable
from uuid import UUID, uuid4

import numpy as np

from .Frame import Frame
from .ABaseNode import ABaseNode


class Input(ABaseNode):
    def __init__(self, name: str):
        ABaseNode.__init__(self, name)
        self.__output = None

    def setOutput(self, output: "Output"):
        self.__output = output

    def getOutput(self) -> "Output":
        return self.__output

    def getDataForFrame(self, frame: Frame) -> np.array:
        return self.getOutput().getDataForFrame(frame)

    def updateAllOutput(self, frame: Frame):
        pass


class Output(ABaseNode):
    def __init__(self, name: str):
        ABaseNode.__init__(self, name)
        self.__computer = None
        self.__data = np.array([])

    def setInitialState(self, initial_state: np.array):
        self.__initial_state = initial_state.copy()

    def reset(self, frame: Frame = None):
        if frame is None:
            frame = Frame(start_timestamp=0, stop_timestamp=0)
        self.setFrame(frame)
        data = self.__initial_state.copy()
        self.setData(data)

    def setComputer(self, computer: "Computer"):
        self.__computer = computer

    def getComputer(self) -> "Computer":
        return self.__computer

    def setData(self, data: np.array):
        self.__data = data

    def getDataForFrame(self, frame: Frame) -> np.array:
        if self.getCurrentFrame() != frame:
            self.setFrame(frame)
            data = self.getComputer().getDataForOuput(frame, self.getID())
            self.setData(data)

        return self.__data

    def updateAllOutput(self, frame: Frame):
        pass


class AComputer(ABaseNode):
    def __init__(self, name: str):
        ABaseNode.__init__(self, name)
        self.__inputs = {}
        self.__outputs = {}

    def isController(self):
        """Checks if the element is derived from AController

        Returns:
          True if the element is derived from AController

        """
        from ..blocks.Controller import AController

        return isinstance(self, AController)

    def setInitialStateForOutput(self, initial_state: np.array, name: str):
        otp = self.getOutputByName(name)
        otp.setInitialState(initial_state)

    def reset(self, frame: Frame = None):
        if frame is None:
            frame = Frame(start_timestamp=0, stop_timestamp=0)
        self.setFrame(frame)
        for oid in self.getListOutputsIds():
            otp = self.getOutputById(oid)
            otp.reset(frame)

    def getListOutputsIds(self) -> Iterable:
        return self.__outputs.keys()

    def getListInputsIds(self) -> Iterable:
        return self.__inputs.keys()

    def defineOutput(self, name: str) -> Output:
        otp = Output(name=name)
        otp.setComputer(self)
        self.__outputs[otp.getID()] = otp
        return otp

    def defineInput(self, name: str) -> Input:
        inp = Input(name)
        self.__inputs[inp.getID()] = inp
        return inp

    def getOutputById(self, output_id: UUID) -> Output:
        return self.__outputs[output_id]

    def getOutputByName(self, name: str) -> Output:
        for oid in self.getListOutputsIds():
            otp = self.getOutputById(oid)
            if otp.getName() == name:
                return otp

        return None

    def getInputById(self, input_id: UUID) -> Input:
        return self.__inputs[input_id]

    def getInputByName(self, name: str) -> Input:
        for iid in self.getListInputsIds():
            inp = self.getInputById(iid)
            if inp.getName() == name:
                return inp

        return None

    def getDataFromInput(
        self, frame: Frame, uid: UUID = None, name: str = None
    ) -> np.array:
        if uid is None and name is None:
            print("[ERROR]")
        elif uid is None and not name is None:
            inp = self.getInputByName(name)
        elif not uid is None and name is None:
            inp = self.getInputById(uid)
        else:
            print("[ERROR]")

        data = inp.getDataForFrame(frame)
        return data

    def getDataForOuput(
        self, frame: Frame, uid: UUID = None, name: str = None
    ) -> np.array:
        if self.getCurrentFrame() != frame:
            self.setFrame(frame)
            self.updateAllOutput(frame)

        if uid is None and name is None:
            print("[ERROR]")
        elif uid is None and not name is None:
            otp = self.getOutputByName(name)
        elif not uid is None and name is None:
            otp = self.getOutputById(uid)
        else:
            print("[ERROR]")

        return otp.getDataForFrame(frame)
