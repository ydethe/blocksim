from collections import defaultdict
import struct
from typing import Iterable
from keyword import kwlist, iskeyword
import logging
from types import FunctionType
from datetime import datetime
import platform
import os
import sys
from numpy.lib.function_base import meshgrid

from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.signal import firwin, fftconvolve
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from . import logger
from .core.Node import AComputer
from .exceptions import *
from .utils import deg, rad
from .dsp.DSPSignal import DSPSignal
from .DatabaseModel import (
    Base,
    Simulation,
    IntegerSerie,
    FloatSerie,
    ComplexSerie,
    BoolSerie,
)


__all__ = ["Logger"]


class Logger(object):
    """Logger to keep track of all variables

    Args:
      fic
         Name of a file to write

    Examples:
      >>> log = Logger('tests/example.log')
      >>> log.openFile()
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
      >>> log2.loadLoggerFile('tests/example.log')
      >>> log2.getValue('2*t')
      array([0., 2.])

    """

    __datetime_fmt = "%Y-%m-%d %H:%M-%S"
    __slots__ = [
        "_dst",
        "_data",
        "_binary",
        "_fic",
        "_mode",
        "__index",
        "__alloc",
        "__defer_write",
        "__start_time",
        "__db_session",
        "__sim",
    ]

    def __init__(self, fic: str = None):
        self._dst = None
        self._fic = None
        self.setOutputLoggerFile(fic)
        self.reset()

    def hasOutputLoggerFile(self) -> bool:
        """Tells if the Logger has an output file

        Returns:
          True if the Logger has an output file

        """
        return not self._fic is None

    def setOutputLoggerFile(self, fic: str, binary: bool = False):
        """Sets the path of the log file

        Args:
          fic
            Path of the log file

        """
        self._binary = binary

        if not fic is None and type(fic) == type(""):
            self._fic = fic
        else:
            return

        if fic.endswith(".pkl") and binary:
            self._fic = fic
            self._mode = "wb"
            self.__defer_write = True
        elif not fic.endswith(".pkl") and binary:
            self._fic = fic
            self._mode = "wb"
            self.__defer_write = False
        elif fic.endswith(".pkl") and not binary:
            raise AssertionError(
                "With .pkl file, you must call Logger.loadLoggerFile with binary=True (fic='%s', binary=%s)"
                % (fic, binary)
            )
        elif not fic.endswith(".pkl") and not binary:
            self._fic = fic
            self._mode = "w"
            self.__defer_write = False

        if fic.startswith("postgresql+psycopg2://"):
            self._fic = fic
            self._mode = ""
            self.__defer_write = True

    def openFile(self):
        self.__index = 0
        if self._fic is None:
            return

        self.__start_time = datetime.now()

        if self._fic.startswith("postgresql+psycopg2://"):
            engine = create_engine(self._fic)
            Base.metadata.bind = engine
            DBSession = sessionmaker(bind=engine)
            self._dst = DBSession()
        else:
            self._dst = open(self._fic, self._mode)

    def allocate(self, size: int):
        self.__alloc = size

    def getFileHeader(self, file: str) -> str:
        f = open(file, mode="rb")
        header = self._load_bin_log_file(stm=f, time_int=None, header_only=True)
        f.close()

        res = "File: %s\n" % header["pth"]
        res += "FileVersion: %s\n" % header["ver"]
        res += "Node: %s\n" % header.get("node", "")
        res += "CreationTime: %s\n" % header.get("creation_time", "")
        res += "User: %s\n" % header.get("user", "")
        res += "BocksimVersion: %s\n" % header.get("blocksim_version", "")

        for cl in header["comm"]:
            res += "%s\n" % cl

        for var in header["variables"]:
            name = var.get("name", "")
            typ = var.get("type", "")
            res += "%s\t%s\n" % (name, typ)

        return res

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
            vals = []
            for x in line.split(","):
                try:
                    vals.append(float(x))
                except ValueError:
                    vals.append(complex(x))

            t = vals[0]
            if time_int is None or time_int[0] <= t and t < time_int[1]:
                for name, val in zip(l_var, vals):
                    self.log(name, val)

            line = stm.readline().strip()

    def _load_bin_log_file(self, stm, time_int, header_only: bool = False):
        sver = stm.read(4)
        if len(sver) != 4:
            raise InvalidLogFile(stm.name)

        ver = struct.unpack("i", sver)[0]

        if ver == 1:
            header = self._load_bin_log_file_v1(stm, time_int, header_only=header_only)
        elif ver == 2:
            header = self._load_bin_log_file_v2(stm, time_int, header_only=header_only)
        elif ver == 3:
            header = self._load_bin_log_file_v3(stm, time_int, header_only=header_only)
        else:
            raise InvalidLogFile(stm.name)

        return header

    def _load_bin_log_file_v1(self, stm, time_int, header_only: bool = False):
        # Lecture entete
        header = {"pth": stm.name}
        sn_var = stm.read(4)
        if len(sn_var) != 4:
            raise InvalidLogFile(stm.name)

        n_var = struct.unpack("i", sn_var)[0]
        header.update({"ver": 1, "comm": [], "variables": []})

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
            var = {"name": name}
            header["variables"].append(var)
            l_var.append(name)

        # Lecture données
        fmt = n_var * "d"
        sze = struct.calcsize(fmt)
        while not header_only:
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

        return header

    def _load_bin_log_file_v2(self, stm, time_int, header_only: bool = False):
        # Lecture entete
        header = {"pth": stm.name}
        sn_var = stm.read(4)
        if len(sn_var) != 4:
            raise InvalidLogFile(stm.name)

        # Lecture de l'entete
        n_var = struct.unpack("i", sn_var)[0]
        header.update({"ver": 2, "comm": [], "variables": []})

        l_var = []
        rec_sze = 0
        for ivar in range(n_var):
            slname = stm.read(4)
            if len(slname) != 4:
                raise InvalidLogFile(stm.name)

            lname = struct.unpack("i", slname)[0]

            bname = stm.read(lname)
            if len(bname) != lname:
                raise InvalidLogFile(stm.name)
            name = bname.decode("utf-8")
            typ = stm.read(1)
            var = {"name": name, "type": typ.decode("utf-8")}
            header["variables"].append(var)
            typ, pck_f, unpck_f, sze = self._findType(typ)
            l_var.append((name, unpck_f, sze))
            rec_sze += sze

        # Lecture données
        while not header_only:
            rec = stm.read(rec_sze)
            if len(rec) == 0:
                break
            elif len(rec) != rec_sze:
                raise InvalidLogFile(stm.name)

            ival = 0
            kval = 0
            for name, unpck_f, sze in l_var:
                val = unpck_f(rec[ival : ival + sze])
                ival += sze
                kval += 1
                if kval == 1:
                    t = val

                if time_int is None or time_int[0] <= t and t < time_int[1]:
                    self.log(name, val)

        return header

    def _load_bin_log_file_v3(self, stm, time_int, header_only: bool = False):
        # Lecture entete
        header = {"pth": stm.name}
        sn_var = stm.read(4)
        if len(sn_var) != 4:
            raise InvalidLogFile(stm.name)

        # Lecture de l'entete
        n_var = struct.unpack("i", sn_var)[0]
        header.update({"ver": 3, "comm": [], "variables": []})
        sn_comm = stm.read(4)
        n_comm = struct.unpack("i", sn_comm)[0]

        b = stm.read(172)
        node = self.__strip_header_txt_line(b)
        b = stm.read(172)
        now = self.__strip_header_txt_line(b)
        b = stm.read(172)
        user = self.__strip_header_txt_line(b)
        b = stm.read(172)
        bs_version = self.__strip_header_txt_line(b)

        header["node"] = node
        header["creation_time"] = now
        header["user"] = user
        header["blocksim_version"] = bs_version

        for _ in range(n_comm - 5):
            b = stm.read(172)
            comm = self.__strip_header_txt_line(b)
            header["comm"].append(comm)

        l_var = []
        rec_sze = 0
        for _ in range(n_var):
            slname = stm.read(4)
            if len(slname) != 4:
                raise InvalidLogFile(stm.name)

            lname = struct.unpack("i", slname)[0]

            bname = stm.read(lname)
            if len(bname) != lname:
                raise InvalidLogFile(stm.name)
            name = bname.decode("utf-8")
            typ = stm.read(1)
            var = {"name": name, "type": typ.decode("utf-8")}
            header["variables"].append(var)
            typ, pck_f, unpck_f, sze = self._findType(typ)
            l_var.append((name, unpck_f, sze))
            rec_sze += sze

        # Lecture données
        while not header_only:
            rec = stm.read(rec_sze)
            if len(rec) == 0:
                break
            elif len(rec) != rec_sze:
                raise InvalidLogFile(stm.name)

            ival = 0
            kval = 0
            for name, unpck_f, sze in l_var:
                val = unpck_f(rec[ival : ival + sze])
                ival += sze
                kval += 1
                if kval == 1:
                    t = val

                if time_int is None or time_int[0] <= t and t < time_int[1]:
                    self.log(name, val)

        return header

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

    @classmethod
    def __strip_header_txt_line(cls, data: bytes) -> str:
        txt = data.decode("utf-8").replace("\x000", "")
        return txt.strip()

    @classmethod
    def __build_header_txt_line(cls, txt: str, length: int = 172) -> bytes:
        b = txt.encode("utf-8")
        n = len(b)
        zf = length - n
        return b + zf * chr(0).encode("utf-8")

    def _update_bin_log_file(self, name):
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
            sver = struct.pack("i", 3)
            self._dst.write(b"%b" % sver)

            llvar = struct.pack("ii", n_var, 4)
            self._dst.write(b"%b" % llvar)

            node = platform.node()
            t0 = datetime.now()
            now = t0.strftime(self.__datetime_fmt)
            try:
                user = os.getlogin()
            except:
                user = "unknown"
            from . import __version__ as bs_version

            bs_version = bs_version

            b = self.__build_header_txt_line(txt=node, length=172)
            self._dst.write(b)

            b = self.__build_header_txt_line(txt=now, length=172)
            self._dst.write(b)

            b = self.__build_header_txt_line(txt=user, length=172)
            self._dst.write(b)

            b = self.__build_header_txt_line(txt=bs_version, length=172)
            self._dst.write(b)

            for var in l_var:
                val = self._data[var][-1]
                typ, _, _, _ = self._findType(val)

                bname = var.encode("utf-8")
                lname = struct.pack("i", len(bname))
                self._dst.write(b"%b%b%b" % (lname, bname, typ))

        # Ecriture du dernier enregistrement
        rec = b""
        for var in l_var:
            val = self._data[var][-1]
            _, pck_f, _, _ = self._findType(val)
            sval = pck_f(val)
            rec += sval

        self._dst.write(rec)

    def _update_ascii_log_file(self, name):
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
        rec = []
        for ivar in range(n_var):
            var = l_var[ivar]
            rec.append(self._data[var][-1])

        fmt = ",".join(["{:.6g}"] * n_var)
        self._dst.write(fmt.format(*rec))
        self._dst.write("\n")

    def _load_postgre(self, uri: str):
        uri, sim_id = uri.split("?sim_id=")
        sim_id = int(sim_id)

        engine = create_engine(uri)
        Base.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        self.__db_session = DBSession()

        q = self.__db_session.query(Simulation).filter(Simulation.id == sim_id)
        self.__sim = q.first()

        q = self.__db_session.query(IntegerSerie.name).filter(
            IntegerSerie.simulation_id == sim_id
        )
        for s in q.all():
            self._data[s.name] = None

        q = self.__db_session.query(FloatSerie.name).filter(
            FloatSerie.simulation_id == sim_id
        )
        for s in q.all():
            self._data[s.name] = None

        q = self.__db_session.query(ComplexSerie.name).filter(
            ComplexSerie.simulation_id == sim_id
        )
        for s in q.all():
            self._data[s.name] = None

        q = self.__db_session.query(BoolSerie.name).filter(
            BoolSerie.simulation_id == sim_id
        )
        for s in q.all():
            self._data[s.name] = None

    def loadLoggerFile(
        self, fic: str, binary: bool = False, time_int: Iterable[float] = None
    ):
        """Loads the content of an existing log file

        Args:
          fic
            Path of a log file

        """
        self._binary = binary

        if fic.endswith(".pkl") and not binary:
            raise AssertionError(
                "With .pkl file, you must call Logger.loadLoggerFile with binary=True (fic='%s', binary=%s)"
                % (fic, binary)
            )

        if binary:
            mode = "rb"
        else:
            mode = "r"

        self.reset()

        if fic.startswith("postgresql+psycopg2://"):
            self._load_postgre(fic)
        elif fic.endswith(".pkl") and binary:
            with open(fic, mode) as f:
                self._data = pd.read_pickle(f)
        elif not fic.endswith(".pkl") and binary:
            with open(fic, mode) as f:
                self._load_bin_log_file(f, time_int)
        elif fic.endswith(".pkl") and not binary:
            raise AssertionError(
                "With .pkl file, you must call Logger.loadLoggerFile with binary=True (fic='%s', binary=%s)"
                % (fic, binary)
            )
        elif not fic.endswith(".pkl") and not binary:
            with open(fic, mode) as f:
                self._load_ascii_log_file(f, time_int)

    def reset(self):
        """Resets the element internal state to zero."""
        self.closeFile()
        self._data = defaultdict(list)
        self.__index = 0
        self.__alloc = 0
        self.__sim = None
        self.__db_session = None

    def export(self, fic: str, format: str):
        for k in self.getParametersName():
            _ = self.getRawValue(k)
        df = pd.DataFrame(self._data)
        print(df)

        if format == "":
            if fic.endswith(".xls"):
                format = "xls"
            elif fic.endswith(".csv"):
                format = "csv"
            elif fic.endswith(".parquet"):
                format = "parquet"
            elif fic.endswith(".feather"):
                format = "feather"

        logger.debug("Export to '%s' (%s)" % (fic, format))

        if format == "xls":
            df.to_excel(fic, engine="openpyxl", na_rep="", index=False)
        elif format == "csv":
            df.to_csv(fic, sep=";", na_rep="", index=False)
        elif format == "parquet":
            df.to_parquet(fic)
        elif format == "feather":
            df.to_feather(fic)

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

        typ_map = {
            b"I": np.int64,
            b"F": np.float64,
            b"C": np.complex128,
            b"B": bool,
        }
        typ, pck_f, unpck_f, sze = self._findType(val)
        dtyp = typ_map[typ]

        if self.__alloc > 0:
            if not name in self._data.keys():
                self._data[name] = np.empty(self.__alloc, dtype=dtyp)
            self._data[name][self.__index] = dtyp(val)
        else:
            self._data[name].append(dtyp(val))

        if name == "t":
            self.__index += 1

        if not self._dst is None and not self.__defer_write:
            if self._binary:
                self._update_bin_log_file(name)
            else:
                self._update_ascii_log_file(name)

    def getParametersName(self) -> Iterable[str]:
        return self._data.keys()

    def getDataSize(self) -> int:
        lnames = list(self.getParametersName())
        if len(lnames) == 0:
            return 0

        name = lnames[0]
        data0 = self.getRawValue(name)

        return len(data0)

    def getFlattenOutput(self, name: str, dtype=np.complex128) -> np.array:
        """Gets the list of output vectors for a computer's output

        Args:
          name
            Name of an output. For example, for a sensor, *sensor_measurement*
          dtype
            Type of the output array

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

    def getValueForComputer(
        self, comp: AComputer, output_name: str, dtype=np.complex128
    ) -> "array":
        """Gets the list of output vectors for a computer's output

        Args:
            comp
                A :class:`blocksim.core.Node.AComputer` whose output are to be retrieved
            output_name
                Name of an output. For example, for a sensor, *measurement*
            dtype
                Type of the output array

        Returns:
            An 2D array of the output

        """
        val = self.getMatrixOutput(
            name="%s_%s" % (comp.getName(), output_name), dtype=dtype
        )
        return val

    def getMatrixOutput(self, name: str) -> np.array:
        """Gets the list of output vectors for a computer's output

        Args:
          name
            Name of an output. For example, for a sensor, *sensor_measurement*

        Returns:
          An 2D array of the output

        """
        lname = []
        for k in self.getParametersName():
            if k.startswith(name):
                lname.append(k)

        nname = len(lname)
        nd = -1
        for idx, name in enumerate(lname):
            data = self.getRawValue(name)
            if nd < 0:
                nd = len(data)
                res = np.empty((nname, nd), dtype=data.dtype)
            res[idx, :] = data

        return res

    def getRawValue(self, name: str) -> np.array:
        """Get the value of a logged variable
        The argument *cannot* be an expression.

        Args:
          name
            Name or expression

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

        if self.__db_session is None:
            data = np.array(self._data[name])
        else:
            data = self.__sim.loadSerieByName(session=self.__db_session, name=name)
        return data

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
        expr = "def __tmp(lg):\n"
        for k in self.getParametersName():
            expr += "   %s=lg.getRawValue('%s')\n" % (k, k)
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
        comp_name = name.replace("_", "")
        return DSPSignal.fromTimeAndSamples(name=comp_name, tps=tps, y_serie=val)

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

    def closeFile(self):
        if self._dst is None:
            return

        sim_id = 0

        if type(self._fic) == type("") and self._fic.endswith(".pkl"):
            df = pd.DataFrame(self._data)
            # df.to_parquet(path=self._dst)
            df.to_pickle(self._dst)
            logger.info("Simulation log saved to '%s'" % os.path.abspath(self._fic))
            self._dst.close()
            self._dst = None
        elif self._fic.startswith("postgresql+psycopg2://"):
            try:
                user = os.getlogin()
            except:
                user = "unknown"

            pgm = list(os.path.split(sys.argv[0]))[-1]
            args = " ".join(sys.argv[1:])
            if pgm.startswith("python"):
                pgm = sys.argv[1]
                args = " ".join(sys.argv[2:])
            sim = Simulation(
                start_time=self.__start_time,
                end_time=datetime.now(),
                program=pgm,
                arguments=args,
                node=platform.node(),
                user=user,
                cwd=os.getcwd(),
            )
            self._dst.add(sim)
            self._dst.commit()
            sim_id = sim.id
            logger.info(
                "Simulation logged in DB. URI %s?sim_id=%i" % (self._fic, sim_id)
            )

            for k in self._data.keys():
                data = self.getRawValue(k)
                typ, pck_f, unpck_f, sze = self._findType(data[0])
                if typ == b"I":
                    s = IntegerSerie(name=k, unit="", data=data, simulation=sim)
                elif typ == b"F":
                    s = FloatSerie(name=k, unit="", data=data, simulation=sim)
                elif typ == b"C":
                    ns = len(data)
                    cdata = np.empty((ns, 2))
                    cdata[:, 0] = np.real(data)
                    cdata[:, 1] = np.imag(data)
                    s = ComplexSerie(name=k, unit="", data=cdata, simulation=sim)
                elif typ == b"B":
                    s = BoolSerie(name=k, unit="", data=data, simulation=sim)
                else:
                    raise ValueError("%s, %s" % (typ, data[0]))
                self._dst.add(s)
            self._dst.commit()

            self._dst = None

        return sim_id
