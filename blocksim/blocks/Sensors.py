from typing import Iterable
from collections import OrderedDict

import numpy as np
from scipy import linalg as lin

from .. import logger
from ..exceptions import *
from ..core.Node import AComputer, AWGNOutput
from ..core.Frame import Frame


__all__ = [
    "ASensors",
    "ProportionalSensors",
    "LinearSensors",
    "StreamSensors",
    "StreamCSVSensors",
]

# TODO : asynchronous sensor


class ASensors(AComputer):
    """Abstract class for a set of sensors

    Implement the method **updateAllOutput** to make it concrete

    The input of the computer is **state**
    The output of the computer is **measurement**

    The parameters mean and cov are to be defined by the user :

    * mean : Mean of the gaussian noise. Dimension (n,1)
    * cov : Covariance of the gaussian noise. Dimension (n,n)
    * cho : Cholesky decomposition of cov, computed after a first call to *updateAllOutput*. Dimension (n,n)

    Args:
      name
        Name of the element
      shape_state
        Shape of the state data
      snames
        Name of each of the scalar components of the measurement.
        Its shape defines the shape of the data

    """

    def __init__(
        self, name: str, shape_state: tuple, snames: Iterable[str], dtype=np.float64
    ):
        AComputer.__init__(self, name)
        self.defineInput("state", shape=shape_state, dtype=dtype)
        otp = AWGNOutput(name="measurement", snames=snames, dtype=dtype)
        otp.setInitialState(np.zeros(otp.getDataShape(), dtype=otp.getDataType()))
        self.addOutput(otp)

        n = otp.getDataShape()[0]
        otp.cov = np.eye(n)
        otp.mean = np.zeros(n)

    def setCovariance(self, cov: np.array):
        otp = self.getOutputByName("measurement")
        otp.cov = cov

    def setMean(self, mean: np.array):
        otp = self.getOutputByName("measurement")
        otp.mean = mean


class ProportionalSensors(ASensors):
    """Models a proportional sensor, which computes the theoritical measurement Y:

    Y = C.X

    with :

    * n number of states X
    * p number of measured states Y

    The input of the computer is **state**
    The output of the computer is **measurement**

    The parameters mean and cov are to be defined by the user :

    * mean : Mean of the gaussian noise. Dimension (n,1)
    * cov : Covariance of the gaussian noise. Dimension (n,n)
    * cho : Cholesky decomposition of cov, computed after a call to *updateAllOutput*. Dimension (n,n)
    * C : Matrix which turns a state into a noiseless measurement

    Args:
      name
        Name of the element
      shape_state
        Number of scalars in the state data
      snames
        Name of each of the scalar components of the measurement.
        Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(
        self, name: str, shape_state: tuple, snames: Iterable[str], dtype=np.float64
    ):
        ASensors.__init__(
            self, name=name, shape_state=shape_state, snames=snames, dtype=dtype
        )

    def updateAllOutput(self, frame: Frame):
        x = self.getDataForInput(frame, name="state")
        meas = self.getOutputByName("measurement")
        meas.setData(self.C @ x)


class LinearSensors(ASensors):
    """Models a linear sensor, which computes the theoritical measurement Y:

    Y = C.X + D.U

    with :

    * n number of states X
    * m number of commands U
    * p number of measured states Y

    The inputs of the element are **state** and **command**
    The output of the computer is **measurement**

    The parameters mean and cov are to be defined by the user :

    * mean : Mean of the gaussian noise. Dimension (n,1)
    * cov : Covariance of the gaussian noise. Dimension (n,n)
    * cho : Cholesky decomposition of cov, computed after a call to *updateAllOutput*. Dimension (n,n)
    * C : (p x n) Output matrix
    * D : (p x m) Feedthrough (or feedforward) matrix

    Args:
      name
        Name of the element
      shape_state
        Number of scalars in the state
      shape_command
        Number of scalars in the command
      snames
        Name of each of the scalar components of the measurement.
        Its shape defines the shape of the data

    """

    def __init__(
        self,
        name: str,
        shape_state: tuple,
        shape_command: tuple,
        snames: Iterable[str],
        dtype=np.float64,
    ):
        ASensors.__init__(
            self, name=name, shape_state=shape_state, snames=snames, dtype=dtype
        )
        self.defineInput("command", shape_command, dtype=dtype)

    def updateAllOutput(self, frame: Frame):
        x = self.getDataForInput(frame, name="state")
        u = self.getDataForInput(frame, name="command")
        meas = self.getOutputByName("measurement")
        meas.setData(self.C @ x + self.D @ u)


class StreamSensors(AComputer):
    """Streams data from a table of values given at initialization

    The element has no inputs
    The output of the computer is **measurement**

    Args:
      name
        Name of the element
      strm_data
        The data. Must be a OrderDict, with a key named 't', and the others determine the name of the outputs.
        Each key must be a np.array with the values of the output variables

    """

    def __init__(self, name: str, strm_data: OrderedDict, dtype=np.float64):
        if not isinstance(strm_data, OrderedDict):
            raise UnorderedDict(self.__class__)

        AComputer.__init__(self, name)

        snames = list(strm_data.keys())[1:]
        otp = self.defineOutput("measurement", snames=snames, dtype=dtype)
        otp.setInitialState(np.zeros(otp.getDataShape(), dtype=otp.getDataType()))

        self.strm_data = strm_data

    def updateAllOutput(self, frame: Frame):
        t2 = frame.getStopTimeStamp()
        meas = self.getOutputByName("measurement")

        val = np.empty(meas.getDataShape(), dtype=meas.getDataType())
        for kv, kn in enumerate(meas.getScalarNames()):
            val[kv] = np.interp(t2, self.strm_data["t"], self.strm_data[kn])

        meas.setData(val)


class StreamCSVSensors(StreamSensors):
    """Streams data from a table of values read in a CSV file

    The inputs of the element are **state** and **command**

    Args:
      name
        Name of the element
      pth
        The path to the CSV file. The file can start with comment lines starting with '#'
        The separator is ','
        The order of the columns matter to define the output vector of the element

    """

    __slots__ = []

    def __init__(self, name: str, pth: str, dtype=np.float64):
        f = open(pth, "r")
        dat = f.readline()
        while dat[0] == "#":
            dat = f.readline()
        f.close()
        elem = dat.strip().split(",")
        strm_data = OrderedDict()
        for kn in elem:
            strm_data[kn] = None

        StreamSensors.__init__(self, name, strm_data, dtype=dtype)

        self.pth = pth

    def resetCallback(self, frame: Frame):
        super().resetCallback(frame)

        f = open(self.pth, "r")
        lines = f.readlines()
        for k in range(len(lines)):
            dat = lines[k]
            if dat[0] != "#":
                nb_dat = len(lines) - k - 1
                break

        self.strm_data["t"] = np.empty(nb_dat, dtype=self.dtype)
        for kn in self.getOutputNames():
            self.strm_data[kn] = np.empty(nb_dat, dtype=self.dtype)

        for j in range(k + 1, len(lines)):
            dat = lines[j]
            elem = dat.strip().split(",")
            t = float(elem[0].strip())
            values = [self.dtype(x.strip()) for x in elem[1:]]
            i = j - k - 1
            self.strm_data["t"][i] = t
            for kv, kn in enumerate(self.getOutputNames()):
                self.strm_data[kn][i] = values[kv]
