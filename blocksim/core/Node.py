from abc import abstractmethod
from typing import Iterable, Iterator, List
from itertools import product
from uuid import UUID, uuid4

import numpy as np
from numpy import sqrt

from .. import logger
from ..utils import assignVector, calc_cho
from ..exceptions import *
from .Frame import Frame
from .ABaseNode import ABaseNode
from .CircularBuffer import CircularBuffer


__all__ = ["Input", "Output", "AWGNOutput", "WeightedFIROutput", "AComputer"]


class Input(ABaseNode):
    """Input node

    The extra attributes  are:
    * __output, which links the `Node.Input` with an `Node.Output`
    * __shape, which is the number of scalars in the data expected by the input
    * __dtype, which is the data type

    Args:
        name: Name of the Input
        shape: Shape of the data expected by the input
        dtype: Data type (typically np.float64 or np.complex128)

    """

    __slots__ = ["__output", "__shape", "__dtype"]

    def __init__(self, name: str, shape: tuple, dtype):
        ABaseNode.__init__(self, name)
        self.__output = None
        if isinstance(shape, int):
            self.__shape = (shape,)
        else:
            self.__shape = shape
        self.__dtype = dtype

    def __repr__(self):
        s = "%s%s" % (self.getName(), self.getDataShape())
        return s

    def setOutput(self, output: "Output"):
        """Sets the output connected to the Input

        Args:
            output: The connected output

        """
        self.__output = output

    def getDataShape(self) -> int:
        return self.__shape

    def getDataType(self) -> int:
        return self.__dtype

    def getOutput(self) -> "Output":
        """Gets the output connected to the Input

        Returns:
            The connected output

        """
        return self.__output

    def getDataForFrame(
        self, frame: Frame, error_on_unconnected: bool = True
    ) -> "array":
        """Gets the data for the given time frame

        Args:
            frame: Time frame of the simulation
            error_on_unconnected: Wether to raise an exception if an unconnected inuput is detected

        Returns:
            The data coming from the connected output

        """
        otp = self.getOutput()

        if error_on_unconnected and otp is None:
            raise SimulationGraphError(
                "The input '%s' is not connected" % self.getName()
            )
        elif not error_on_unconnected and otp is None:
            data = np.zeros(self.getDataShape(), dtype=self.getDataType())
            oname = "[unconnected]"
        else:
            data = otp.getDataForFrame(frame, error_on_unconnected=error_on_unconnected)
            oname = otp.getName()

        valid_data = assignVector(
            data, self.getDataShape(), self.getName(), oname, self.getDataType()
        )

        if self.getCurrentFrame() != frame and frame.getTimeStep() == 0:
            self.resetCallback(frame)

        return valid_data

    def updateAllOutput(self, frame: Frame):
        """Unused for an input"""
        pass


class Output(ABaseNode):
    """Output node

    Args:
        name: Name of the Output
        snames: Name of each of the scalar components of the data.
            Its shape defines the shap of the data
        dtype: Data type (typically np.float64 or np.complex128)

    """

    __slots__ = [
        "__computer",
        "__shape",
        "__dtype",
        "__data",
        "__snames",
        "__initial_state",
    ]

    def __init__(self, name: str, snames: Iterable[str], dtype=np.float64):
        ABaseNode.__init__(self, name)
        self.__computer = None
        self.__data = np.array([])
        self.__dtype = dtype
        self.__snames = np.array(snames)
        self.__shape = self.__snames.shape

    def __repr__(self):
        s = "%s%s" % (self.getName(), self.getDataShape())
        return s

    def setInitialState(self, initial_state: "array"):
        """Sets the element's initial state vector

        Args:
            initial_state: The element's initial state vector

        """
        valid_data = assignVector(
            initial_state,
            self.getDataShape(),
            self.getName(),
            "<arg>",
            self.getDataType(),
        )
        self.__initial_state = valid_data

    def getScalarNames(self) -> Iterable[str]:
        """Gets the name of each of the scalar components of the data

        Returns:
            The name of each of the scalar components of the data

        """
        return self.__snames

    def iterScalarNameValue(
        self, frame: Frame, error_on_unconnected: bool = True
    ) -> Iterator:
        """Iterate through all the data, and yield the name and the value of the scalar

        Yields:
            The next tuple of name and the value of the scalar

        """
        ns = self.getDataShape()
        dat = self.getDataForFrame(frame, error_on_unconnected=error_on_unconnected)

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in ns:
            it.append(range(k))

        # Iterate over all dimensions
        for iscal in product(*it):
            yield self.__snames[iscal], dat[iscal]

    def getInitialeState(self) -> "array":
        """Gets the element's initial state vector

        Returns:
            The element's initial state vector

        """
        return self.__initial_state

    def getDataShape(self) -> tuple:
        return self.__shape

    def getDataType(self):
        return self.__dtype

    def setComputer(self, computer: "Computer"):
        """Sets the computer containing the Output

        Args:
            computer: The computer to be set

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
            data: The data for the output

        """
        comp = self.getComputer()
        nc = comp.getName()
        no = self.getName()
        valid_data = assignVector(
            data, self.getDataShape(), "%s.%s" % (nc, no), "<arg>", self.__dtype
        )
        self.__data = valid_data

    def getDataForFrame(
        self, frame: Frame, error_on_unconnected: bool = True
    ) -> "array":
        """Gets the data for the Output at the given time frame
        If the given time frame is different from the last one seen by the Output,
        the update of the simulation is triggered.

        Args:
            frame: The time frame
            error_on_unconnected: Wether to raise an exception if an unconnected inuput is detected

        """
        if self.getCurrentFrame() != frame:
            self.setFrame(frame)
            if frame.getTimeStep() == 0:
                self.resetCallback(frame)
                self.setData(self.getInitialeState())
            data = self.getComputer().getDataForOutput(
                frame, self.getID(), error_on_unconnected=error_on_unconnected
            )
            self.setData(data)

        return self.__data

    def updateAllOutput(self, frame: Frame):
        """Unused for Output"""
        pass


class AWGNOutput(Output):

    __slots__ = ["mean", "cov", "cho", "cplxe"]

    def resetCallback(self, frame: Frame):
        """Resets the element internal state to zero."""
        if (
            self.mean.shape[0] != self.cov.shape[0]
            or self.mean.shape[0] != self.cov.shape[1]
        ):
            raise ValueError(
                "[ERROR]Bad dimensions for the covariance. %s instead of %s"
                % (str(self.cov.shape), str((self.mean.shape[0], self.mean.shape[0])))
            )

        self.cho = calc_cho(self.cov)

        typ = self.getDataType()
        if typ == np.complex128 or typ == np.complex64:
            self.cplxe = True
        else:
            self.cplxe = False

    def addGaussianNoise(self, state: "array") -> "array":
        """Adds a gaussian noise to a state vector

        Args:
            state: State vector without noise

        Returns:
            Vector of noisy measurements

        """
        if self.cplxe:
            bn = np.random.normal(size=len(state)) * 1j / sqrt(2)
            bn += np.random.normal(size=len(state)) / sqrt(2)
        else:
            bn = np.random.normal(size=len(state))

        bg = self.cho.T @ bn + self.mean

        return state + bg

    def getDataForFrame(
        self, frame: Frame, error_on_unconnected: bool = True
    ) -> "array":
        if self.getCurrentFrame() != frame:
            self.setFrame(frame)
            if frame.getTimeStep() == 0:
                self.setData(self.getInitialeState())
                self.resetCallback(frame)
            data = self.getComputer().getDataForOutput(
                frame, self.getID(), error_on_unconnected=error_on_unconnected
            )
            noisy = self.addGaussianNoise(data)
            self.setData(noisy)

        return super().getDataForFrame(frame, error_on_unconnected=error_on_unconnected)


class WeightedFIROutput(Output):
    """To be useable, the AComputer that uses WeightedFIROutput
    has to implement a generateCoefficients method that returns
    the taps weighting the output.
    See `dsp.DSPFilter.BandpassDSPFilter.generateCoefficients`

    """

    __slots__ = ["__yprev", "__a_taps", "__a_buf", "__b_taps", "__b_buf"]

    def __init__(self, name: str, snames: List[str], dtype):
        Output.__init__(self, name=name, snames=snames, dtype=dtype)
        self.setInitialState(initial_state=np.array([0], dtype=dtype))

    def resetCallback(self, frame: Frame):
        filt = self.getComputer()
        typ = self.getDataType()

        x0 = self.getInitialeState()
        self.__yprev = x0[0]

        self.__b_taps, self.__a_taps = filt.generateCoefficients()
        na = len(self.__a_taps)
        nb = len(self.__b_taps)
        self.__a_buf = CircularBuffer(size=na, dtype=typ)
        self.__b_buf = CircularBuffer(size=nb, dtype=typ)

    def processSample(self, sample: np.complex128) -> np.complex128:
        self.__a_buf.append(self.__yprev)
        self.__b_buf.append(sample)
        na = len(self.__a_buf)
        nb = len(self.__b_buf)
        ba = self.__a_buf.getAsArray()
        bb = self.__b_buf.getAsArray()
        a0 = self.__a_taps[0]
        ya = ba[na - 1 : 0 : -1] @ self.__a_taps[1:]
        xb = bb[::-1] @ self.__b_taps
        y = (xb - ya) / a0
        self.__yprev = y
        return y


class AComputer(ABaseNode):
    """Abstract class for all the computers of the control chain.
    A AComputer contains a list of `Node.Input`
    and a list of `Node.Output`

    Implement **compute_outputs** to make it concrete

    Args:
        name: Name of the element
        logged: True to log the computer in the Simulation's log

    Examples:
        >>> e = DummyComputer(name='tst')

    """

    __slots__ = ["__inputs", "__outputs", "__parameters", "__logged"]

    def __init__(self, name: str, logged: bool = True):
        ABaseNode.__init__(self, name)
        self.__inputs = {}
        self.__outputs = {}
        self.__parameters = {}
        self.__logged = logged

    @property
    def isLogged(self):
        return self.__logged

    def __repr__(self):
        s = ""
        sn = "'%s'" % self.getName()
        sc = self.__class__.__name__
        tot_w = 2 + len(sn)
        tot_w = max(tot_w, len(sc) + 2)

        s_out = []
        for otp in self.getListOutputs():
            s_out.append(str(otp))

        s_inp = []
        for inp in self.getListInputs():
            s_inp.append(str(inp))

        if len(s_out) == 0:
            out_w = 2
        else:
            out_w = 2 + max([len(s) for s in s_out])

        if len(s_inp) == 0:
            inp_w = 2
        else:
            inp_w = 2 + max([len(s) for s in s_inp])

        tot_w = max(tot_w, out_w - 2 + inp_w)

        s += "   =" + tot_w * "=" + "=\n"
        s += "   |" + sn.center(tot_w) + "|\n"
        s += "   |" + sc.center(tot_w) + "|\n"
        s += "   =" + tot_w * "=" + "=\n"
        s += "   |" + tot_w * " " + "|\n"

        for k in range(max(len(s_out), len(s_inp))):
            if k < len(s_inp):
                xin = s_inp[k]
                pre = "-> "
            else:
                xin = "|"
                pre = "   "

            if k < len(s_out):
                xout = s_out[k]
                post = " ->"
            else:
                xout = "|"
                post = "   "

            s += pre + xin.ljust(inp_w) + xout.rjust(out_w) + post + "\n"
            s += "   |" + tot_w * " " + "|\n"

        s += "   =" + tot_w * "=" + "=\n"

        return s

    def createParameter(self, name: str, value: float = None, read_only: bool = False):
        """This method creates an attribute, with getter an optional setter
        Use `AComputer.printParameters` to see the list of all declared parameters

        Args:
            name: Name of the parameter to be created
            value: Value of the parameter to be created
            read_only: If True, the value cannot be modified (no setter defined)

        Examples:
            >>> e = DummyComputer('el')
            >>> e.createParameter('val', 0)
            >>> e.val
            0
            >>> e.val = 2
            >>> e.val
            2
            >>> e.createParameter('ro_val', 1, read_only=True)
            >>> e.ro_val
            1

        """
        self.__parameters[name] = value

        def get(self):
            return self.__parameters[name]

        if read_only:
            setattr(self.__class__, name, property(get))
        else:

            def set(self, val):
                self.__parameters[name] = val

            setattr(self.__class__, name, property(get, set))

    def printParameters(self) -> str:
        """Prints the list of all declared parameters and their values
        Paremeter declaration is made through `AComputer.createParameter`

        Returns:
            A string containing the parameters and their value

        """
        s = ""
        sn = "'%s'" % self.getName()
        sc = self.__class__.__name__
        tot_w = 2 + len(sn)
        tot_w = max(tot_w, len(sc) + 2)

        s_out = []
        for otp in self.getListOutputs():
            s_out.append(str(otp))

        s_inp = []
        for inp in self.getListInputs():
            s_inp.append(str(inp))

        if len(s_out) == 0:
            out_w = 2
        else:
            out_w = 2 + max([len(s) for s in s_out])

        if len(s_inp) == 0:
            inp_w = 2
        else:
            inp_w = 2 + max([len(s) for s in s_inp])

        tot_w = max(tot_w, out_w - 2 + inp_w)

        s += "=" + tot_w * "=" + "=\n"
        s += "|" + sn.center(tot_w) + "|\n"
        s += "|" + sc.center(tot_w) + "|\n"
        s += "=" + tot_w * "=" + "=\n"

        for k in self.__parameters.keys():
            val = self.__parameters[k]
            l = "%s:\t%s\n" % (k, val)
            s += l

        return s

    def getValueFromLogger(
        self, logger: "Logger", output_name: str, dtype=np.complex128
    ) -> "array":
        """Gets the list of output vectors for a computer's output

        Args:
            logger: A `Logger.Logger` that contains the values
            output_name
                Name of an output. For example, for a sensor, *measurement*
            dtype
                Type of the output array

        Returns:
            An 2D array of the output

        """
        val = logger.getMatrixOutput(
            name="%s_%s" % (self.getName(), output_name), dtype=dtype
        )
        return val

    def isController(self) -> bool:
        """Checks if the element is derived from AController
        See `control.Controller.AController`

        Returns:
          True if the element is derived from AController

        """
        from ..control.Controller import AController

        return isinstance(self, AController)

    def setInitialStateForOutput(self, initial_state: np.array, output_name: str):
        """Sets the initial state vector for a given output

        Args:
            initial_state: The initial state vector
            output_name: The output's initial state vector

        """
        otp = self.getOutputByName(output_name)
        otp.setInitialState(initial_state)

    def getInitialStateForOutput(self, output_name: str) -> "array":
        """Sets the initial state vector for a given output

        Args:
            output_name: The output's initial state vector

        Returns:
            The initial state vector

        """
        otp = self.getOutputByName(output_name)
        return otp.getInitialeState()

    def getListOutputs(self) -> Iterable[Output]:
        """Gets the list of the outputs

        Returns:
            The list of outputs

        """
        return self.__outputs.values()

    def getListOutputsIds(self) -> Iterable[UUID]:
        """Gets the list of the outputs' ids

        Returns:
            The list of UUID

        """
        return self.__outputs.keys()

    def getListOutputsNames(self) -> Iterable[str]:
        """Gets the list of the outputs' names

        Returns:
            The list of names

        """
        for oid in self.getListOutputsIds():
            otp = self.getOutputById(oid)
            yield otp.getName()

    def getListInputs(self) -> Iterable[Input]:
        """Gets the list of the inputs

        Returns:
            The list of inputs

        """
        return self.__inputs.values()

    def getListInputsIds(self) -> Iterable[UUID]:
        """Gets the list of the inputs' ids

        Returns:
            The list of UUID

        """
        return self.__inputs.keys()

    def getListInputsNames(self) -> Iterable[str]:
        """Gets the list of the inputs' names

        Returns:
            The list of names

        """
        for iid in self.getListInputsIds():
            inp = self.getInputById(iid)
            yield inp.getName()

    def defineOutput(self, name: str, snames: Iterator[str], dtype) -> Output:
        """Creates an output for the computer

        Args:
            name: Name of the output
            snames: Name of each of the scalar components of the data.
                Its shape defines the shap of the data
            dtype: Data type (typically np.float64 or np.complex128)

        Returns:
            The created output

        """
        otp = Output(name=name, snames=snames, dtype=dtype)
        otp.setInitialState(np.zeros(otp.getDataShape(), dtype=otp.getDataType()))
        otp.setComputer(self)
        self.addOutput(otp)
        return otp

    def addOutput(self, otp: Output):
        """Adds an output for the computer

        Args:
            otp: Output to add

        """
        if otp.getName() in self.getListOutputsNames():
            raise DuplicateOutput(self.getName(), otp.getName())

        otp.setComputer(self)
        self.__outputs[otp.getID()] = otp

    def replaceOutput(self, old_name: str, new_output: Output):
        """Replaces an output for the computer

        Args:
            old_name: Name of the output te replace
            new_output: new output that replaces the old one

        """
        otp = self.getOutputByName(old_name)
        oid = otp.getID()
        del self.__outputs[oid]
        self.addOutput(new_output)

    def removeOutput(self, oname: str):
        """Removes the specified Output

        Args:
            oname: The name of the Output to remove

        """
        otp = self.getOutputByName(oname)
        otp.setComputer(None)
        oid = otp.getID()
        self.__outputs.pop(oid)

    def defineInput(self, name: str, shape: int, dtype) -> Input:
        """Creates an input for the computer

        Args:
            name: Name of the input
            shape: Shape of the data expected by the input
            dtype: Data type (typically np.float64 or np.complex128)

        Returns:
            The created input

        """
        inp = Input(name, shape=shape, dtype=dtype)
        self.addInput(inp)
        return inp

    def addInput(self, inp: Input):
        """Adds an input for the computer

        Args:
            inp: Input to add

        """
        if inp.getName() in self.getListInputsNames():
            raise DuplicateInput(self.getName(), inp.getName())

        self.__inputs[inp.getID()] = inp

    def replaceInput(self, old_name: str, new_input: Input):
        """Replaces an input for the computer

        Args:
            old_name: Name of the input te replace
            new_input: new input that replaces the old one

        """
        inp = self.getInputByName(old_name)
        iid = inp.getID()
        del self.__inputs[iid]
        self.addInput(new_input)

    def removeInput(self, iname: str):
        """Removes the specified Input

        Args:
            iname: The name of the Input to remove

        """
        inp = self.getInputByName(iname)
        iid = inp.getID()
        self.__inputs.pop(iid)

    def getOutputById(self, output_id: UUID) -> Output:
        """Get an output with its id

        Args:
            output_id: Id of the output to retreive

        Returns:
            The Output

        """
        if not output_id in self.__outputs.keys():
            logger.error("In '%s' : id not found '%s'" % (self.getName(), output_id))
            raise UnknownOutput(self.getName(), output_id)
        return self.__outputs[output_id]

    def getOutputByName(self, name: str) -> Output:
        """Get an output with its name

        Args:
            name: Name of the output to retreive

        Returns:
            The Output

        """
        for oid in self.getListOutputsIds():
            otp = self.getOutputById(oid)
            if otp.getName() == name:
                return otp

        logger.error("In '%s' : output name not found '%s'" % (self.getName(), name))
        raise UnknownOutput(self.getName(), name)

    def getInputById(self, input_id: UUID) -> Input:
        """Get an input with its id

        Args:
            input_id: Id of the input to retreive

        Returns:
            The Input

        """
        if not input_id in self.__inputs.keys():
            logger.error("In '%s' : id not found '%s'" % (self.getName(), input_id))
            raise UnknownInput(self.getName(), input_id)
        return self.__inputs[input_id]

    def getInputByName(self, name: str) -> Input:
        """Get an input with its name

        Args:
            name: Name of the input to retreive

        Returns:
            The Input

        """
        for iid in self.getListInputsIds():
            inp = self.getInputById(iid)
            if inp.getName() == name:
                return inp

        logger.error("In '%s' : input name not found '%s'" % (self.getName(), name))
        raise UnknownInput(self.getName(), name)

    def getDataForInput(
        self,
        frame: Frame,
        uid: UUID = None,
        name: str = None,
        error_on_unconnected: bool = True,
    ) -> "array":
        """Gets the data for the given input, either withs its id or with its name
        One and only one of uid and name shall be given

        Args:
            frame: The time frame
            uid: The id of the input
            name: The name of the input

        Returns:
            The data

        """
        if uid is None and name is None:
            logger.error(
                "In '%s' : Unable to find input id=%s, name='%s'"
                % (self.getName(), uid, name)
            )
        elif uid is None and not name is None:
            inp = self.getInputByName(name)
        elif not uid is None and name is None:
            inp = self.getInputById(uid)
        else:
            logger.error(
                "In '%s' : Unable to find input id=%s, name='%s'"
                % (self.getName(), uid, name)
            )

        data = inp.getDataForFrame(frame, error_on_unconnected=error_on_unconnected)
        return data

    def getDataForOutput(
        self,
        frame: Frame,
        uid: UUID = None,
        name: str = None,
        error_on_unconnected: bool = True,
    ) -> "array":
        """Gets the data for the given output, either withs its id or with its name
        One and only one of uid and name shall be given

        Args:
            frame: The time frame
            uid: The id of the output
            name: The name of the output
            error_on_unconnected: Wether to raise an exception if an unconnected inuput is detected

        Returns:
            The data

        """
        if self.getCurrentFrame() != frame:
            self.setFrame(frame)
            self.updateAllOutput(frame, error_on_unconnected=error_on_unconnected)

        if uid is None and name is None:
            logger.error(
                "In '%s' : Unable to find output id=%s, name='%s'"
                % (self.getName(), uid, name)
            )
        elif uid is None and not name is None:
            otp = self.getOutputByName(name)
        elif not uid is None and name is None:
            otp = self.getOutputById(uid)
        else:
            logger.error(
                "In '%s' : Unable to find output id=%s, name='%s'"
                % (self.getName(), uid, name)
            )

        return otp.getDataForFrame(frame, error_on_unconnected=error_on_unconnected)

    def getParents(self) -> Iterable["AComputer"]:
        """Returns the parent computers

        Returns:
            An iterator on the computers' parents

        """
        for iid in self.getListInputsIds():
            inp = self.getInputById(iid)
            otp = inp.getOutput()
            c = otp.getComputer()
            yield c

    @abstractmethod
    def compute_outputs(self, **inputs: dict) -> dict:  # pragma: no cover
        pass

    def updateAllOutput(self, frame: Frame, error_on_unconnected: bool = True):
        """Updates all the outputs of the Computer

        Args:
            frame: The time frame
            error_on_unconnected: Wether to raise an exception if an unconnected inuput is detected

        """
        inputs = {}
        for inp in self.getListInputs():
            k = inp.getName()
            v = inp.getDataForFrame(frame, error_on_unconnected=error_on_unconnected)
            inputs[k] = v
        for otp in self.getListOutputs():
            k = otp.getName()
            v = otp.getDataForFrame(frame, error_on_unconnected=error_on_unconnected)
            inputs[k] = v
        inputs["t1"] = frame.getStartTimeStamp()
        inputs["t2"] = frame.getStopTimeStamp()

        outputs = self.compute_outputs(**inputs)

        for otp in self.getListOutputs():
            k = otp.getName()
            otp.setData(outputs[k])


class DummyComputer(AComputer):

    __slots__ = []

    def __init__(self, name: str, with_input: bool = True):
        AComputer.__init__(self, name)
        if with_input:
            self.defineInput("xin", shape=1, dtype=np.int64)
        self.defineOutput("xout", snames=["x"], dtype=np.int64)
        self.setInitialStateForOutput(np.array([0]), output_name="xout")

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        **inputs,
    ) -> dict:
        outputs = {}
        outputs["xout"] = np.array([0])

        return outputs
