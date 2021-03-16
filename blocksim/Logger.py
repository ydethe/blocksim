from collections import defaultdict
import struct
from typing import Iterable
from keyword import kwlist, iskeyword
import logging
from types import FunctionType

import numpy as np
from scipy.signal import firwin, fftconvolve

from .exceptions import *
from .utils import deg, rad
from .dsp.DSPSignal import DSPSignal


__all__ = ["Logger"]


class Logger(object):
    """Logger to keep track of all variables

    Args:
      fic
         Name of a file to write

    Examples:
      >>> log = Logger('example.log')
      >>> log.hasOutputLoggerFile()
      True
      >>> log.log('t',0)
      >>> log.log('t',1)
      >>> log.getValue('t')
      array([0, 1])
      >>> log.getValue('2*t')
      array([0, 2])
      >>> del log
      >>> log2 = Logger()
      >>> log2.loadLoggerFile('example.log')
      >>> log2.getValue('2*t')
      array([0., 2.])

    """

    __slots__ = ["_dst", "_data", "_binary"]

    def __init__(self, fic: str = None):
        self._dst = None
        self.setOutputLoggerFile(fic)
        self.reset()

    def hasOutputLoggerFile(self) -> bool:
        """Tells if the Logger has an output file

        Returns:
          True if the Logger has an output file

        """
        return not self._dst is None

    def setOutputLoggerFile(self, fic: str, binary: bool = False):
        """Sets the path of the log file

        Args:
          fic
            Path of the log file

        """
        self._binary = binary
        if binary:
            mode = "wb"
        else:
            mode = "w"

        if not fic is None and type(fic) == type(""):
            self._dst = open(fic, mode)

    def _load_ascii_log_file(self, stm, time_int):
        ver = int(stm.readline().strip())

        if ver == 1:
            self._load_ascii_log_file_v1(stm, time_int)
        else:
            raise InvalidLogFile(stm.name)

    def _load_ascii_log_file_v1(self, stm, time_int):
        # Lecture entete
        n_var = int(stm.readline().strip())

        l_var = []
        for ivar in range(n_var):
            var = stm.readline().strip()
            l_var.append(var)

        # Lecture données
        line = stm.readline().strip()
        while line != "":
            vals = [float(x) for x in line.split(",")]

            t = vals[0]
            if time_int is None or time_int[0] <= t and t < time_int[1]:
                for name, val in zip(l_var, vals):
                    self.log(name, val)

            line = stm.readline().strip()

    def _load_bin_log_file(self, stm, time_int):
        sver = stm.read(4)
        if len(sver) != 4:
            raise InvalidLogFile(stm.name)

        ver = struct.unpack("i", sver)[0]

        if ver == 1:
            self._load_bin_log_file_v1(stm, time_int)
        else:
            raise InvalidLogFile(stm.name)

    def _load_bin_log_file_v1(self, stm, time_int):
        # Lecture entete
        sn_var = stm.read(4)
        if len(sn_var) != 4:
            raise InvalidLogFile(stm.name)

        n_var = struct.unpack("i", sn_var)[0]

        l_var = []
        for ivar in range(n_var):
            slname = stm.read(4)
            if len(slname) != 4:
                raise InvalidLogFile(stm.name)

            lname = struct.unpack("i", slname)[0]

            bname = stm.read(lname)
            if len(bname) != lname:
                raise InvalidLogFile(stm.name)
            name = bname.decode("utf-8")
            l_var.append(name)

        # Lecture données
        fmt = n_var * "d"
        sze = struct.calcsize(fmt)
        while True:
            rec = stm.read(sze)
            if len(rec) == 0:
                break
            elif len(rec) != sze:
                raise InvalidLogFile(stm.name)
            vals = struct.unpack(fmt, rec)

            t = vals[0]
            if time_int is None or time_int[0] <= t and t < time_int[1]:
                for name, val in zip(l_var, vals):
                    self.log(name, val)

    def _update_bin_log_file(self, name, val):
        if name != "t":
            return

        n = len(self._data["t"])
        l_var = list(self._data.keys())
        n_var = len(l_var)
        ivart = l_var.index("t")
        l_var.pop(ivart)
        l_var.insert(0, "t")

        # Ecriture de l'entete si pas déjà fait
        if n == 1:
            sver = struct.pack("i", 1)
            self._dst.write(b"%b" % sver)

            llvar = struct.pack("i", n_var)
            self._dst.write(b"%b" % llvar)

            for var in l_var:
                bname = var.encode("utf-8")
                lname = struct.pack("i", len(bname))
                self._dst.write(b"%b%b" % (lname, bname))

        # Ecriture du dernier enregistrement
        rec = b""
        for var in l_var:
            val = self._data[var][-1]
            sval = struct.pack("d", val)
            rec += sval

        self._dst.write(rec)

    def _update_ascii_log_file(self, name, val):
        if name != "t":
            return

        n = len(self._data["t"])
        l_var = list(self._data.keys())
        n_var = len(l_var)
        ivart = l_var.index("t")
        l_var.pop(ivart)
        l_var.insert(0, "t")

        # Ecriture de l'entete si pas déjà fait
        if n == 1:
            self._dst.write("1\n")
            self._dst.write("%i\n" % n_var)

            for var in l_var:
                self._dst.write("%s\n" % var)

        # Ecriture du dernier enregistrement
        rec = np.empty(n_var)
        for ivar in range(n_var):
            var = l_var[ivar]
            rec[ivar] = self._data[var][-1]

        self._dst.write(",".join(["%f"] * n_var) % tuple(rec))
        self._dst.write("\n")

    def loadLoggerFile(
        self, fic: str, binary: bool = False, time_int: Iterable[float] = None
    ):
        """Loads the content of an existing log file

        Args:
          fic
            Path of a log file

        """
        self._binary = binary
        if binary:
            mode = "rb"
        else:
            mode = "r"

        try:
            f = open(fic, mode)
        except Exception as e:
            raise FileError(fic)

        self.reset()

        if binary:
            self._load_bin_log_file(f, time_int)
        else:
            self._load_ascii_log_file(f, time_int)

        f.close()

    def reset(self):
        """Resets the element internal state to zero."""
        self._data = defaultdict(list)
        if not self._dst is None:
            self._dst.truncate(0)

    def log(self, name: str, val: float):
        """Log a value for a variable. If *name* is '_', nothing is logged

        Args:
          name
            Name of the parameter. Nothing is logged if *name* == '_'
          val
            Value to log

        """
        if name == "_":
            return

        if iskeyword(name) or name == "keep_up":
            raise NameIsPythonKeyword(name)

        self._data[name].append(val)

        if not self._dst is None:
            if self._binary:
                self._update_bin_log_file(name, val)
            else:
                self._update_ascii_log_file(name, val)

    def getValue(self, name: str) -> np.array:
        """Get the value of a logged variable
        The argument can be an expression. It can combine several variables
        numpy functions can be used with the module name 'np': for example : np.cos

        Args:
          name
            Name or expression

        Returns:
          An array of the values

        Examples:
          >>> log = Logger()
          >>> _ = [log.log('a',a) for a in np.linspace(0,2*np.pi,200)]
          >>> r = log.getValue('np.cos(a)**2 + np.sin(a)**2')
          >>> np.max(np.abs(r-1)) < 1e-15
          True

        """
        if len(self._data.keys()) == 0:
            raise SystemError(u"[ERROR]Logger empty")

        expr = "def __tmp(lg):\n"
        for k in self._data.keys():
            expr += "   %s=np.array(lg._data['%s'])\n" % (k, k)
        expr += "   return %s" % name

        foo_code = compile(expr, "<string>", "exec")
        foo_func = FunctionType(foo_code.co_consts[0], globals(), "__tmp")

        return foo_func(self)

    def getSignal(self, name: str) -> DSPSignal:
        """Get the value of a logged variable
        The argument can be an expression. It can combine several variables
        numpy functions can be used with the module name 'np': for example : np.cos

        Args:
          name
            Name or expression

        Returns:
          A :class:`blocksim.dsp.DSPSignal.DSPSignal`

        Examples:
          >>> log = Logger()
          >>> _ = [log.log('t',a) for a in np.linspace(0,2*np.pi,200)]
          >>> sig = log.getSignal('np.cos(t)**2 + np.sin(t)**2')
          >>> np.max(np.abs(sig.y_serie-1)) < 1e-15
          True

        """
        tps = self.getValue("t")
        val = self.getValue(name)
        return DSPSignal.fromTimeAndSamples(name=name, tps=tps, y_serie=val)

    def getFilteredValue(
        self, name: str, ntaps: int, cutoff: float, window: str = "hamming"
    ) -> np.array:
        """Get the value of a logged variable, and applies a low-pass filter
        The argument can be an expression. It can combine several variables
        numpy functions can be used with the module name 'np': for example : np.cos

        Args:
          name
            Name or expression
          ntaps
            Number of coefficients of the filter
          cutoff
            Cutoff frequency, normalized by the Nyquist frequency
          window
            Window function to apply. Default : hamming

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

    def __del__(self):
        if not self._dst is None:
            self._dst.close()
