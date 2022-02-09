from typing import Iterable
from datetime import datetime
import platform
import os
import sys

from ..LoggerSpec import if_suitable
import pluggy
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..Logger import Logger
from .. import logger
from ..exceptions import *
from ..DatabaseModel import (
    Base,
    Simulation,
    IntegerSerie,
    FloatSerie,
    ComplexSerie,
    BoolSerie,
)

__all__ = ["Logger"]

hookimpl = pluggy.HookimplMarker("blocksim")


class Logger(object):
    @hookimpl
    def test_suitable(self, logger: Logger) -> bool:
        uri = logger.getLoadedFile()
        if uri is None:
            return False

        istat = uri.startswith("postgresql+psycopg2://")
        return istat

    @if_suitable
    @hookimpl
    def loadLogFile(self, logger: Logger):
        """Loads the content of an existing log file

        Args:
          fic
            Path of a log file

        """
        uri = logger.getLoadedFile()
        uri, sim_id = uri.split("?sim_id=")
        sim_id = int(sim_id)

        engine = create_engine(uri)
        Base.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        self.__db_session = DBSession()

        q = self.__db_session.query(Simulation).filter(Simulation.id == sim_id)
        self.__sim = q.first()

        for k in self.__sim.listSeriesNames(self.__db_session):
            logger.createEmptyValue(k)

    @if_suitable
    @hookimpl
    def getRawValue(self, logger: Logger, name: str) -> "array":
        lnames = logger.getParametersName()
        if len(lnames) == 0:
            raise SystemError("Logger empty")
        if not name in lnames:
            raise SystemError("Logger has no variable '%s'" % name)

        data = logger.__sim.loadSerieByName(session=logger.__db_session, name=name)

        return data

    @if_suitable
    @hookimpl
    def export(self, logger: Logger):
        uri = logger.getLoadedFile()
        engine = create_engine(uri)
        Base.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        self.__db_session = DBSession()

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
            start_time=logger.getStartTime(),
            end_time=datetime.now(),
            program=pgm,
            arguments=args,
            node=platform.node(),
            user=user,
            cwd=os.getcwd(),
        )
        self.__db_session.add(sim)
        self.__db_session.commit()
        sim_id = sim.id
        logger.info("Simulation logged in DB. URI %s?sim_id=%i" % (uri, sim_id))

        for k in logger.getParametersName():
            data = self.getRawValue(k)
            typ, pck_f, unpck_f, sze = logger._findType(data[0])
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
            self.__db_session.add(s)
        self.__db_session.commit()

        self.__db_session = None
