from typing import Iterable
from uuid import UUID, uuid4

import numpy as np

from ..exceptions import *
from .Frame import Frame
from .ABaseNode import ABaseNode
from .. import logger
from ..utils import assignVector


class Input(ABaseNode):
    """Input node

    The extra attributes  are:
    * __output, which links the :class:`blocksim.core.Node.Input` with an :class:`blocksim.core.Node.Output`
    * __nscal, which is the number of scalars in the data expected by the input
    * __dtype, which is the data type

    Args:
      name
        Name of the Input
      nscal
        Number of scalars in the data expected by the input
      dtype
        Data type (typically np.float64 or np.complex128)

    """

    def __init__(self, name: str, nscal: int, dtype):
        ABaseNode.__init__(self, name)
        self.__output = None
        self.__nscal = nscal
        self.__dtype = dtype

    def setOutput(self, output: "Output"):
        """Sets the output connected to the Input

        Args:
          output
            The connected output

        """
        self.__output = output

    def getNumberScalar(self) -> int:
        return self.__nscal

    def getOutput(self) -> "Output":
        """Gets the output connected to the Input

        Returns:
            The connected output

        """
        return self.__output

    def getDataForFrame(self, frame: Frame) -> np.array:
        """Gets the data for the given time frame

        Returns:
            The data coming from the connected output

        """
        otp = self.getOutput()
        data = otp.getDataForFrame(frame)
        valid_data = assignVector(
            data, self.getNumberScalar(), self.getName(), otp.getName(), self.__dtype
        )
        return valid_data

    def updateAllOutput(self, frame: Frame):
        """Unused for an input"""
        pass


class Output(ABaseNode):
    """Output node

    The extra attribute are :

    * __computer, which contains the output
    * __data, which contains the data communicated to the connected Inputs
    * __nscal, which is the number of scalars in the data expected by the input
    * __dtype, which is the data type

    Args:
      name
        Name of the Output
      nscal
        Number of scalars in the data expected by the input
      dtype
        Data type (typically np.float64 or np.complex128)

    """

    def __init__(self, name: str, nscal: int, dtype=np.float64):
        ABaseNode.__init__(self, name)
        self.__computer = None
        self.__data = np.array([])
        self.__nscal = nscal
        self.__dtype = dtype

    def setInitialState(self, initial_state: np.array):
        """Sets the element's initial state vector

        Args:
          initial_state
            The element's initial state vector

        """
        valid_data = assignVector(
            initial_state, self.getNumberScalar(), self.getName(), "<arg>", self.__dtype
        )
        self.__initial_state = valid_data

    def reset(self, frame: Frame = None):
        """Resets the element internal state the value given by :class:`blocksim.core.Output.setInitialState`"""
        if frame is None:
            frame = Frame(start_timestamp=0, stop_timestamp=0)
        self.setFrame(frame)
        self.__data = self.__initial_state.copy()

    def getNumberScalar(self) -> int:
        return self.__nscal

    def setComputer(self, computer: "Computer"):
        """Sets the computer containing the Output

        Args:
          computer
            The computer to be set

        """
        self.__computer = computer

    def getComputer(self) -> "Computer":
        """Gets the computer containing the Output

        Returns:
            The computer set for the output

        """
        return self.__computer

    def setData(self, data: np.array):
        """Sets the data for the Output

        Args:
          data
            The data for the output

        """
        valid_data = assignVector(
            data, self.getNumberScalar(), self.getName(), "<arg>", self.__dtype
        )
        self.__data = valid_data

    def getDataForFrame(self, frame: Frame) -> np.array:
        """Gets the data for the Output at the given time frame
        If the given time frame is different from the last one seen by the Output,
        the update of the simulation is triggered.

        Args:
          frame
            The time frame

        """
        if self.getCurrentFrame() != frame:
            self.setFrame(frame)
            data = self.getComputer().getDataForOutput(frame, self.getID())
            self.setData(data)

        return self.__data

    def updateAllOutput(self, frame: Frame):
        """Unused for Output"""
        pass


class AComputer(ABaseNode):
    """Abstract class for all the computers of the control chain.
    A AComputer contains a list of :class:`blocksim.blocks.Node.Input`
    and a list of :class:`blocksim.blocks.Node.Output`

    Args:
      name
        Name of the element

    Examples:
      >>> e = DummyComputer(name='tst')

    """

    def __init__(self, name: str):
        ABaseNode.__init__(self, name)
        self.__inputs = {}
        self.__outputs = {}

    def isController(self) -> bool:
        """Checks if the element is derived from AController
        See :class:`blocksim.blocks.Controller.AController`

        Returns:
          True if the element is derived from AController

        """
        from ..blocks.Controller import AController

        return isinstance(self, AController)

    def setInitialStateForOutput(self, initial_state: np.array, output_name: str):
        """Sets the initial state vector for a given output

        Args:
          initial_state
            The initial state vector
          output_name
            The output's initial state vector

        """
        otp = self.getOutputByName(output_name)
        otp.setInitialState(initial_state)

    def reset(self, frame: Frame = None):
        """Resets the computer's outputs state

        Args:
          frame
            Time frame for reset

        Examples:
          >>> e = DummyComputer('el')
          >>> frame=Frame()
          >>> e.reset(frame)

        """
        if frame is None:
            frame = Frame(start_timestamp=0, stop_timestamp=0)
        self.setFrame(frame)
        for oid in self.getListOutputsIds():
            otp = self.getOutputById(oid)
            otp.reset(frame)

    def getListOutputsIds(self) -> Iterable[UUID]:
        """Gets the list of the outputs' ids

        Returns:
          The list of UUID

        """
        return self.__outputs.keys()

    def getListInputsIds(self) -> Iterable[UUID]:
        """Gets the list of the inputs' ids

        Returns:
          The list of UUID

        """
        return self.__inputs.keys()

    def defineOutput(self, name: str, nscal: int, dtype) -> Output:
        """Creates an output for the computer

        Args:
          name
            Name of the output
          nscal
            Number of scalars in the data expected by the input
          dtype
            Data type (typically np.float64 or np.complex128)

        Returns:
          The created output

        """
        otp = Output(name=name, nscal=nscal, dtype=dtype)
        otp.setComputer(self)
        self.__outputs[otp.getID()] = otp
        return otp

    def defineInput(self, name: str, nscal: int, dtype) -> Input:
        """Creates an input for the computer

        Args:
          name
            Name of the input
          nscal
            Number of scalars in the data expected by the input
          dtype
            Data type (typically np.float64 or np.complex128)

        Returns:
          The created input

        """
        inp = Input(name, nscal=nscal, dtype=dtype)
        self.__inputs[inp.getID()] = inp
        return inp

    def getOutputById(self, output_id: UUID) -> Output:
        """Get an output with its id

        Args:
          output_id
            Id of the output to retreive

        Returns:
          The Output

        """
        return self.__outputs[output_id]

    def getOutputByName(self, name: str) -> Output:
        """Get an output with its name

        Args:
          name
            Name of the output to retreive

        Returns:
          The Output

        """
        for oid in self.getListOutputsIds():
            otp = self.getOutputById(oid)
            if otp.getName() == name:
                return otp

        return None

    def getInputById(self, input_id: UUID) -> Input:
        """Get an input with its id

        Args:
          input_id
            Id of the input to retreive

        Returns:
          The Input

        """
        return self.__inputs[input_id]

    def getInputByName(self, name: str) -> Input:
        """Get an input with its name

        Args:
          name
            Name of the input to retreive

        Returns:
          The Input

        """
        for iid in self.getListInputsIds():
            inp = self.getInputById(iid)
            if inp.getName() == name:
                return inp

        return None

    def getDataForInput(
        self, frame: Frame, uid: UUID = None, name: str = None
    ) -> np.array:
        """Gets the data for the given input, either withs its id or with its name
        One and only one of uid and name shall be given

        Args:
          frame
            The time frame
          uid
            The id of the input
          name
            The name of the input

        Returns:
          The data

        """
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

    def getDataForOutput(
        self, frame: Frame, uid: UUID = None, name: str = None
    ) -> np.array:
        """Gets the data for the given output, either withs its id or with its name
        One and only one of uid and name shall be given

        Args:
          frame
            The time frame
          uid
            The id of the output
          name
            The name of the output

        Returns:
          The data

        """
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

    def getParents(self) -> Iterable["AComputer"]:
        """Returns the parent computers

        Returns:
          The parent computers

        """
        res = []
        for iid in self.getListInputsIds():
            inp = self.getInputById(iid)
            otp = inp.getOutput()
            c = otp.getComputer()
            res.append(c)

        return res


class DummyComputer(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("in")
        self.defineOutput("out")
        self.setInitialStateForOutput(np.array([0]), output_name="out")

    def updateAllOutput(self, frame: Frame):
        pass
