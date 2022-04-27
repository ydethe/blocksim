"""Implementation of Logger

"""

from collections import defaultdict
from typing import Iterable
from keyword import iskeyword
from types import FunctionType
from datetime import datetime, timezone
import os
from pathlib import Path

import pluggy
import numpy as np
from numpy import *
from scipy.signal import firwin, fftconvolve

from .. import logger
from ..core.Node import AComputer
from ..exceptions import *
from ..utils import deg, rad
from ..dsp.DSPSignal import DSPSignal
from .. import plugin_manager


__all__ = ["Logger"]

hookimpl = pluggy.HookimplMarker("blocksim")


class Logger(object):
    """Logger to keep track of all variables

    Examples:
        >>> log = Logger()
        >>> log.log('t',0)
        >>> log.log('t',1)
        >>> log.getValue('t') # doctest: +ELLIPSIS
        array([0, 1]...
        >>> log.getValue('2*t') # doctest: +ELLIPSIS
        array([0, 2]...
        >>> log.export('tests/example.csv')
        >>> del log
        >>> log2 = Logger()
        >>> log2.loadLogFile('tests/example.csv')
        >>> log2.getValue('2*t') # doctest: +ELLIPSIS
        array([0, 2]...

    """

    __datetime_fmt = "%Y-%m-%d %H:%M-%S"
    __slots__ = [
        "__data",
        "__uri",
        "__index",
        "__alloc",
        "__start_time",
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the logger and empties all the recorded data"""
        self.__uri = None
        self.__start_time = datetime.now(tz=timezone.utc)
        self.__data = defaultdict(list)
        self.__index = 0
        self.__alloc = -1

    def createEmptyValue(self, name: str):
        """Creates a new variable with no data

        Args:
            name: The name of the variable

        """
        self.__data[name] = None

    def setRawData(self, data: dict):
        """Sets the data for the whole logger.
        The dictionary data shall have one key per variable to create,
        and the associated value shall be an array of values.

        Args:
            data: Data to use

        """
        self.__data = data

    def getRawData(self) -> dict:
        """Gets the data for the whole logger.
        The dictionary data has one key per variable to create,
        and the associated value is an array of values.

        Returns:
            The dictionary of the data

        """
        return self.__data

    def setStartTime(self, t: datetime):
        """Sets the start time

        Args:
            t: the start time

        """
        self.__start_time = t

    def getStartTime(self) -> datetime:
        """Gets the start time

        Returns:
            the start time

        """
        return self.__start_time

    def getLoadedFile(self) -> Path:
        """Get the name of the file or the URI associated to the logger

        Returns:
            The path or URI

        """
        return self.__uri

    def loadLogFile(self, uri: Path):
        """Loads a specified file or URI into the logger.
        Looks up among all plugins which one can handle the extension of **fig**

        Args:
            uri: The path or URI where the data will be written

        """
        self.__uri = Path(uri)
        if not self.__uri.exists():
            raise FileNotFoundError(self.__uri)

        ldata = plugin_manager.hook.loadLogFile(log=self, uri=self.__uri)
        lok = [x for x in ldata if x]
        if len(lok) == 0:
            raise IOError("No logger to handle '%s'" % self.__uri)
        elif len(lok) > 1:
            raise IOError("Too many loggers to handle '%s'" % self.__uri)

    def allocate(self, size: int):
        """If called with a size, pre allocates the data with an array of size *size*

        Args:
            size: Size to pre allocate

        """
        self.__alloc = size

    @classmethod
    def _findType(cls, val):
        from struct import pack, unpack, calcsize

        def unpack_cplxe(z):
            x, y = unpack("dd", z)
            return x + 1j * y

        fmt_dict = {b"I": "q", b"F": "d", b"C": "dd", b"B": "?"}

        pck_dict = {
            b"I": lambda x: pack(fmt_dict[b"I"], x),
            b"F": lambda x: pack(fmt_dict[b"F"], x),
            b"C": lambda x: pack(fmt_dict[b"C"], np.real(x), np.imag(x)),
            b"B": lambda x: pack(fmt_dict[b"B"], x),
        }

        unpck_dict = {
            b"I": lambda x: unpack("q", x)[0],
            b"F": lambda x: unpack("d", x)[0],
            b"C": unpack_cplxe,
            b"B": lambda x: unpack("?", x)[0],
        }

        if isinstance(val, bytes):
            typ = val
        elif isinstance(val, (int, np.int8, np.int16, np.int32, np.int64)):
            typ = b"I"
        elif isinstance(val, (float, np.float16, np.float32, np.float64)):
            typ = b"F"
        elif isinstance(val, (complex, np.complex64, np.complex128)):
            typ = b"C"
        elif isinstance(val, (bool)):
            typ = b"B"
        else:
            raise ValueError("Impossible to determine type of %s" % val)

        pck_f = pck_dict[typ]
        unpck_f = unpck_dict[typ]
        sze = calcsize(fmt_dict[typ])
        return typ, pck_f, unpck_f, sze

    def log(self, name: str, val: float):
        """Log a value for a variable. If *name* is '_', nothing is logged

        Args:
            name: Name of the parameter. Nothing is logged if *name* == '_'
            val: Value to log

        """
        if name == "_":
            return

        if iskeyword(name) or name == "keep_up":
            raise NameIsPythonKeyword(name)

        typ_map = {
            b"I": np.int64,
            b"F": np.float64,
            b"C": np.complex128,
            b"B": bool,
        }
        typ, pck_f, unpck_f, sze = self._findType(val)
        dtyp = typ_map[typ]

        if self.__alloc > 0:
            if not name in self.__data.keys():
                self.__data[name] = np.empty(self.__alloc, dtype=dtyp)
            self.__data[name][self.__index] = dtyp(val)
        else:
            self.__data[name].append(dtyp(val))

        if name == "t":
            self.__index += 1

        # Handling of the logged value by the pugins
        ldata = plugin_manager.hook.log(log=self, name=name, val=val)

    def getParametersName(self) -> Iterable[str]:
        """Returns an iterable on all the variables of the logger

        Returns:
            An iterator over all the variables of the logger

        """
        return self.__data.keys()

    def getDataSize(self) -> int:
        """Returns the size of the data stored in the logger.
        This is computed by taking the size of the value array of the first variable.

        Returns:
            The size of the data stored in the logger

        """
        lnames = list(self.getParametersName())
        if len(lnames) == 0:
            return 0

        name = lnames[0]
        data0 = self.getRawValue(name)

        return len(data0)

    def getFlattenOutput(self, name: str, dtype=np.complex128) -> "array":
        """Gets the list of output vectors for a computer's output

        Args:
            name: Name of an output. For example, for a sensor, *sensor_measurement*
            dtype: Type of the output array

        Returns:
            An 1D array of the output

        """
        lname = []
        for k in self.getParametersName():
            if k.startswith(name):
                lname.append(k)

        nname = len(lname)
        nd = self.getDataSize()
        res = np.empty(nd * nname, dtype=dtype)
        for idx, name in enumerate(lname):
            res[idx::nname] = self.getValue(name)

        return res

    def getValueForComputer(self, comp: AComputer, output_name: str) -> "array":
        """Gets the list of output vectors for a computer's output

        Args:
            comp: A `blocksim.core.Node.AComputer` whose output are to be retrieved
            output_name: Name of an output. For example, for a sensor, *measurement*

        Returns:
            An 2D array of the output

        """
        val = self.getMatrixOutput(name="%s_%s" % (comp.getName(), output_name))
        return val

    def getMatrixOutput(self, name: str) -> "array":
        """Gets the list of output vectors for a computer's output

        Args:
            name: Name of an output. For example, for a sensor, *sensor_measurement*

        Returns:
            An 2D array of the output

        """
        lname = []
        for k in self.getParametersName():
            if k.startswith(name):
                lname.append(k)

        nname = len(lname)
        if nname == 0:
            raise KeyError(name)

        nd = -1
        for idx, name in enumerate(lname):
            data = self.getRawValue(name)
            if nd < 0:
                nd = len(data)
                res = np.empty((nname, nd), dtype=data.dtype)
            res[idx, :] = data

        return res

    def getRawValue(self, name: str) -> "array":
        """Get the value of a logged variable
        The argument *cannot* be an expression.

        Args:
            name: Name or expression

        Returns:
            An array of the values

        Examples:
            >>> log = Logger()
            >>> ref = np.linspace(0,2*np.pi,200)
            >>> _ = [log.log('a',a) for a in ref]
            >>> r = log.getRawValue('a')
            >>> np.max(np.abs(r-ref)) < 1e-15
            True

        """
        lnames = self.getParametersName()
        if len(lnames) == 0:
            raise SystemError("Logger empty")
        if not name in lnames:
            raise SystemError("Logger has no variable '%s'" % name)

        ldata = plugin_manager.hook.getRawValue(log=self, name=name)
        lok = [x for x in ldata if not x is None]
        if len(lok) == 1:
            return lok[0]
        elif len(lok) == 0:
            data = self.getRawData()
            return np.array(data[name])
        else:
            raise ValueError(
                "Too many loggers to handle file '%s'" % self.getLoadedFile()
            )

    def getValue(self, name: str) -> "array":
        """Get the value of a logged variable
        The argument can be an expression. It can combine several variables
        numpy functions can be used with the module name 'np': for example : np.cos

        Args:
            name: Name or expression

        Returns:
            An array of the values

        Examples:
            >>> log = Logger()
            >>> _ = [log.log('a',a) for a in np.linspace(0,2*np.pi,200)]
            >>> r = log.getValue('np.cos(a)**2 + np.sin(a)**2')
            >>> np.max(np.abs(r-1)) < 1e-15
            True

        """
        lnames = self.getParametersName()
        if len(lnames) == 0:
            raise SystemError("Logger empty")

        expr = "def __tmp(lg):\n"
        for k in self.getParametersName():
            expr += "   %s=lg.getRawValue(name='%s')\n" % (k, k)
        expr += "   return %s" % name

        foo_code = compile(expr, "<string>", "exec")
        foo_func = FunctionType(foo_code.co_consts[0], globals(), "__tmp")

        return foo_func(self)

    def getSignal(self, name: str) -> DSPSignal:
        """Get the value of a logged variable
        The argument can be an expression. It can combine several variables
        numpy functions can be used with the module name 'np': for example : np.cos

        Args:
            name: Name or expression

        Returns:
            A `blocksim.dsp.DSPSignal.DSPSignal`

        Examples:
            >>> log = Logger()
            >>> _ = [log.log('t',a) for a in np.linspace(0,2*np.pi,200)]
            >>> sig = log.getSignal('np.cos(t)**2 + np.sin(t)**2')
            >>> np.max(np.abs(sig.y_serie-1)) < 1e-15
            True

        """
        tps = self.getValue("t")
        val = self.getValue(name)
        comp_name = name.replace("_", "")
        return DSPSignal.fromTimeAndSamples(name=comp_name, tps=tps, y_serie=val)

    def getFilteredValue(
        self, name: str, ntaps: int, cutoff: float, window: str = "hamming"
    ) -> "array":
        """Get the value of a logged variable, and applies a low-pass filter
        The argument can be an expression. It can combine several variables
        numpy functions can be used with the module name 'np': for example : np.cos

        Args:
            name: Name or expression
            ntaps: Number of coefficients of the filter
            cutoff: Cutoff frequency, normalized by the Nyquist frequency
            window: Window function to apply. Default : hamming

        Returns:
            An array of the values

        Examples:
            >>> ns=1000;fs=100;f0=10
            >>> log = Logger()
            >>> _ = [(log.log('t',t),log.log('s',np.cos(np.pi*2*t*f0))) for t in np.arange(0,ns)/fs]
            >>> s = log.getValue('s')
            >>> e = np.sum(s**2)/ns
            >>> e # doctest: +ELLIPSIS
            0.5...
            >>> sf = log.getFilteredValue('s', ntaps=64, cutoff=20*2/fs) # 20 Hz filter
            >>> ef = np.sum(sf**2)/ns
            >>> ef # doctest: +ELLIPSIS
            0.498...
            >>> sf = log.getFilteredValue('s', ntaps=64, cutoff=5*2/fs) # 5 Hz filter
            >>> ef = np.sum(sf**2)/ns
            >>> ef # doctest: +ELLIPSIS
            4.6...e-05

        """
        sig = self.getValue(name)
        a = firwin(numtaps=ntaps, cutoff=cutoff, window=window)
        y = fftconvolve(sig, a, mode="same")
        return y

    def export(self, uri: Path):
        """Exports the logger to a specified file or URI into the logger.
        Looks up among all plugins which one can handle the extension of **fig**

        Args:
            uri: The path or URI to write in

        Raises:
            SystemError if no handler has been identified or if too many handlers were identified

        """
        self.__uri = Path(uri)

        lstat = plugin_manager.hook.export(log=self, uri=self.__uri)

        lok = [x for x in lstat if x >= 0]

        if len(lok) == 0:
            raise SystemError("Unable to write '%s'" % self.__uri)
        elif len(lok) > 1:
            raise SystemError("Uncoherent return '%s'" % lok)