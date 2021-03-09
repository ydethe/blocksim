from abc import ABCMeta, abstractmethod
from typing import Iterable
from uuid import UUID, uuid4

import numpy as np


class Frame(object):
    def __init__(self, start_timestamp: float = 0, stop_timestamp: float = 0):
        self.__start_timestamp = start_timestamp
        self.__stop_timestamp = stop_timestamp
        self.__id = uuid4()

    def __repr__(self):
        s = "<%s %s> start=%g, stop=%g <end>" % (
            self.__class__.__name__,
            self.getFrameID(),
            self.getStartTimeStamp(),
            self.getStopTimeStamp(),
        )
        return s

    def getStartTimeStamp(self) -> float:
        return self.__start_timestamp

    def getStopTimeStamp(self) -> float:
        return self.__stop_timestamp

    def getTimeStep(self) -> float:
        return self.__stop_timestamp - self.__start_timestamp

    def getFrameID(self) -> UUID:
        return self.__id

    def updateByStep(self, step: float):
        if step == 0:
            return

        self.__start_timestamp = self.__stop_timestamp
        self.__stop_timestamp += step
        self.__id = uuid4()

    def copy(self) -> "Frame":
        res = Frame(
            start_timestamp=self.getStartTimeStamp(),
            stop_timestamp=self.getStopTimeStamp(),
        )
        res.__id = self.getFrameID()
        return res

    def __eq__(self, y: "Frame") -> bool:
        return self.getFrameID() == y.getFrameID()


class ABaseNode(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.__name = name
        self.__id = uuid4()
        self.__current_frame = None
        self.__data = np.array([])

    def getName(self):
        return self.__name

    def getID(self) -> UUID:
        return self.__id

    def setData(self, data: np.array):
        self.__data = data

    def getCurrentFrame(self) -> UUID:
        return self.__current_frame

    def setFrame(self, frame: Frame):
        self.__current_frame = frame.copy()

    @abstractmethod
    def updateAllOutput(self, frame: Frame):
        pass


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
    def __init__(self, name: str, initial_state: np.array):
        ABaseNode.__init__(self, name)
        self.__computer = None
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

    def defineOutput(self, name: str, initial_state: np.array) -> Output:
        otp = Output(name=name, initial_state=initial_state)
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

    def getDataFromInput(self, frame: Frame, input_id: UUID) -> np.array:
        inp = self.getInputById(input_id)
        data = inp.getDataForFrame(frame)
        return data

    def getDataForOuput(self, frame: Frame, output_id: UUID) -> np.array:
        if self.getCurrentFrame() != frame:
            self.setFrame(frame)
            self.updateAllOutput(frame)

        otp = self.getOutputById(output_id)

        return otp.getDataForFrame(frame)


def connect(
    computer_src: AComputer, output_name: str, computer_dst: AComputer, intput_name: str
):
    otp = computer_src.getOutputByName(output_name)
    inp = computer_dst.getInputByName(intput_name)
    inp.setOutput(otp)
