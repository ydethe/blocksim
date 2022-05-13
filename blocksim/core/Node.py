from abc import abstractmethod
from typing import Iterable, Iterator, List
from itertools import product
from uuid import UUID, uuid4

import numpy as np
from numpy import sqrt

from .. import logger
from ..utils import assignVector, calc_cho
from ..exceptions import *
from .ABaseNode import ABaseNode
from .CircularBuffer import CircularBuffer


__all__ = ["Input", "Output", "AWGNOutput", "TFOutput", "AComputer"]


class Input(ABaseNode):
    """Input node

    Args:
        name: Name of the Input
        shape: Shape of the data expected by the input
        dtype: Data type (typically np.float64 or np.complex128)

    """

    __slots__ = ["__shape", "__dtype"]

    def __init__(self, name: str, shape: tuple, dtype):
        ABaseNode.__init__(self, name)
        if isinstance(shape, int):
            self.__shape = (shape,)
        else:
            self.__shape = shape
        self.__dtype = dtype

    def getDefaultInputData(self) -> "array":
        return np.zeros(self.getDataShape(), dtype=self.getDataType())

    def __repr__(self):
        s = "%s%s" % (self.getName(), self.getDataShape())
        return s

    def getDataShape(self) -> int:
        return self.__shape

    def getDataType(self) -> int:
        return self.__dtype

    def process(self, data: "array") -> "array":
        """Applies a transform to the incoming data.
        By default, does nothing.

        Args:
            data: The array to work on

        Returns:
            The transformed array. Must be the same shape as the input

        """
        return data


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
        "__tdata",
        "__snames",
        "__initial_state",
    ]

    def __init__(self, name: str, snames: Iterable[str], dtype=np.float64):
        ABaseNode.__init__(self, name)
        self.__computer = None
        self.__data = np.array([])
        self.__tdata = np.array([])
        self.__dtype = dtype
        self.__snames = np.array(snames)
        self.__shape = self.__snames.shape
        self.setInitialState(np.zeros(self.__shape, dtype=dtype))

    def setComputer(self, comp: "AComputer"):
        if not self.__computer is None:
            if self.__computer.getName() != comp.getName():
                logger.warning(
                    f"Replacing AComputer associated with Output '{self.getName()}': '{self.__computer.getName()}' to '{comp.getName()}'"
                )

        self.__computer = comp

    def getComputer(self) -> "AComputer":
        return self.__computer

    def __repr__(self):
        s = "%s%s" % (self.getName(), self.getDataShape())
        return s

    def resetCallback(self, t0: float):
        super().resetCallback(t0)
        dat = self.getInitialeState()
        self.setData(dat)

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

    def iterScalarNameValue(self) -> Iterator:
        """Iterate through all the data, and yield the name and the value of the scalar

        Yields:
            The next tuple of name and the value of the scalar

        """
        ns = self.getDataShape()

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in ns:
            it.append(range(k))

        # Iterate over all dimensions
        for iscal in product(*it):
            yield self.__snames[iscal], self.__tdata[iscal]

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

    def setData(self, data: "array", cname: str = "?"):
        """Sets the data for the Output
        data is the data **before** applying `Output.process`

        Args:
            data: The data for the output

        """
        no = self.getName()
        valid_data = assignVector(
            data, self.getDataShape(), "%s.%s" % (cname, no), "<arg>", self.__dtype
        )
        self.__data = valid_data
        self.__tdata = self.process(valid_data)

    def _getUnprocessedData(
        self,
    ) -> "array":
        """Gets the data for the Output

        Returns:
            data: The data for the output

        """
        return self.__data.copy()

    def getData(
        self,
    ) -> "array":
        """Gets the data for the Output

        Returns:
            data: The data for the output

        """
        return self.__tdata.copy()

    def process(self, data: "array") -> "array":
        """Applies a transform to the outgoing data.
        By default, does nothing.
        The **transformed** data is stored

        Args:
            data: The array to work on

        Returns:
            The transformed array. Must be the same shape as the input

        """
        return data


class AWGNOutput(Output):

    __slots__ = ["seed", "mean", "cov", "cho", "cplxe"]

    def __init__(self, name: str, snames: Iterable[str], dtype=np.float64):
        super().__init__(name, snames, dtype)
        self.seed = 46351657

        if dtype == np.complex128 or dtype == np.complex64:
            self.cplxe = True
        else:
            self.cplxe = False

    def resetCallback(self, t0: float):
        np.random.seed(self.seed)

        if (
            self.mean.shape[0] != self.cov.shape[0]
            or self.mean.shape[0] != self.cov.shape[1]
        ):
            raise ValueError(
                "[ERROR]Bad dimensions for the covariance. %s instead of %s"
                % (str(self.cov.shape), str((self.mean.shape[0], self.mean.shape[0])))
            )

        self.cho = calc_cho(self.cov)

        super().resetCallback(t0)

    def process(self, state: "array") -> "array":
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


class TFOutput(Output):
    """To be useable, the AComputer that uses TFOutput
    has to implement a generateCoefficients method that returns
    the taps weighting the output.
    See `blocksim.dsp.DSPFilter.ADSPFilter.generateCoefficients`

    Args:
        name: Name of the output
        snames: List of the names of the outputs
        dtype: Type of the output

    """

    __slots__ = ["__yprev", "__a_taps", "__a_buf", "__b_taps", "__b_buf"]

    def __init__(self, name: str, snames: List[str], dtype):
        Output.__init__(self, name=name, snames=snames, dtype=dtype)
        self.setInitialState(initial_state=np.array([0], dtype=dtype))

    def resetCallback(self, t0: float):
        filt = self.getComputer()
        typ = self.getDataType()

        x0 = self.getInitialeState()
        self.__yprev = x0[0]

        self.__b_taps, self.__a_taps = filt.generateCoefficients()
        na = len(self.__a_taps)
        nb = len(self.__b_taps)
        self.__a_buf = CircularBuffer(size=na, dtype=typ)
        self.__b_buf = CircularBuffer(size=nb, dtype=typ)

        super().resetCallback(t0)

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

    def process(self, data: "array") -> "array":
        res = np.empty_like(data)
        rs = res.shape
        li = []
        for d in rs:
            li.append(range(d))
        for ind in product(*li):
            res[ind] = self.processSample(data[ind])

        return res


class AComputer(ABaseNode):
    """Abstract class for all the computers of the control chain.
    A AComputer contains a list of `Input`
    and a list of `Output`

    Implement **update** to make it concrete

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
            logger: A `blocksim.loggers.Logger.Logger` that contains the values
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

    def resetCallback(self, t0: float):
        super().resetCallback(t0)
        for otp in self.getListOutputs():
            otp.resetCallback(t0)
        for itp in self.getListInputs():
            itp.resetCallback(t0)

    def isController(self) -> bool:
        """Checks if the element is derived from AController
        See `blocksim.control.Controller.AController`

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

    def defineOutput(self, name: str, snames: List[str], dtype) -> Output:
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
        self.addOutput(otp)
        otp.setComputer(self)
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
        new_output.setComputer(self)
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

    def getDataForOutput(self, oname: str) -> "array":
        otp = self.getOutputByName(name=oname)
        return otp.getData()

    @abstractmethod
    def update(self, **inputs: dict) -> dict:  # pragma: no cover
        """Method used to update a Node.

        Args:
            t1: Current simulation time (s)
            t2: New simulation time (s)

        """
        pass


class DummyComputer(AComputer):

    __slots__ = []

    def __init__(self, name: str, with_input: bool = True):
        AComputer.__init__(self, name)
        if with_input:
            self.defineInput("xin", shape=1, dtype=np.int64)
        self.defineOutput("xout", snames=["x"], dtype=np.int64)
        self.setInitialStateForOutput(np.array([0]), output_name="xout")

    def update(
        self,
        t1: float,
        t2: float,
        **inputs,
    ) -> dict:
        outputs = {}
        outputs["xout"] = np.array([0])

        return outputs
