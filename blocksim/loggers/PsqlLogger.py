from typing import Iterable
from datetime import datetime
import platform
import os
import sys

import numpy as np
import pluggy
import pandas as pd
from singleton3 import Singleton
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .Logger import Logger
from ..exceptions import *
from ..db.DatabaseModel import (
    Base,
    Simulation,
    IntegerSerie,
    FloatSerie,
    ComplexSerie,
    BoolSerie,
)

__all__ = ["Logger"]

hookimpl = pluggy.HookimplMarker("blocksim")


class Logger(object, metaclass=Singleton):
    @hookimpl
    def test_suitable(self, uri: str) -> bool:
        if uri is None:
            return False

        istat = uri.parts[0] == "postgresql+psycopg2:"
        return istat

    @hookimpl
    def loadLogFile(self, log: "Logger", uri: str):
        if not self.test_suitable(uri):
            return False

        uri, sim_id = uri.split("?sim_id=")
        sim_id = int(sim_id)

        engine = create_engine(uri)
        Base.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        self.__db_session = DBSession()

        q = self.__db_session.query(Simulation).filter(Simulation.id == sim_id)
        self.__sim = q.first()

        for k in self.__sim.listSeriesNames(self.__db_session):
            log.createEmptyValue(k)

        return True

    @hookimpl
    def getRawValue(self, log: "Logger", name: str) -> "array":
        uri = log.getLoadedFile()
        if not self.test_suitable(uri):
            return

        data = self.__sim.loadSerieByName(session=self.__db_session, name=name)

        return data

    @hookimpl
    def export(self, log: "Logger", uri: str) -> int:
        from .. import logger

        if not self.test_suitable(uri):
            return -1

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
            start_time=log.getStartTime(),
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

        raw_data = log.getRawData()
        for k in log.getParametersName():
            data = raw_data[k]
            typ, pck_f, unpck_f, sze = log._findType(data[0])
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

        return sim_id

    @hookimpl
    def log(self, log: "Logger", name: str, val: float) -> int:
        return 1
