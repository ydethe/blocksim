from typing import Any
import os
import json
import pickle

from nptyping import NDArray
import pluggy
import pandas as pd
from singleton3 import Singleton

from .Parameter import Parameter
from .Logger import Logger
from .. import logger
from ..exceptions import *


__all__ = ["Logger"]

hookimpl = pluggy.HookimplMarker("blocksim")


class Logger(object, metaclass=Singleton):
    @hookimpl
    def test_suitable(self, uri: str) -> bool:
        if uri is None:
            return False

        istat = uri.endswith(".pkl")
        return istat

    @hookimpl
    def loadLogFile(self, log: "blocksim.loggers.Logger.Logger", uri: str):
        if not self.test_suitable(uri):
            return False

        if not os.path.exists(uri):
            raise FileNotFoundError(uri)

        log.reset()
        with open(uri, "rb") as f:
            pdesc, data = pickle.load(f)

        for name, unit, desc, styp in pdesc:
            typ = log.typ_map[styp.encode("utf-8")]
            log.createEmptyValue(name=name, unit=unit, description=desc, dtype=typ)
        log.setRawData(data)
        return True

    @hookimpl
    def getRawValue(
        self, log: "blocksim.loggers.Logger.Logger", name: str
    ) -> NDArray[Any, Any]:
        return

    @hookimpl
    def export(self, log: "blocksim.loggers.Logger.Logger", uri: str) -> int:
        if not self.test_suitable(uri):
            return -1

        pdesc = []
        for p in log.getParameters():
            styp = p.getTypeDB()

            if styp == "complex":
                typ = "C"
            elif styp == "integer":
                typ = "I"
            elif styp == "float":
                typ = "F"
            elif styp == "boolean":
                typ = "B"
            pdesc.append(
                (
                    p.name,
                    p.unit,
                    p.description,
                    typ,
                )
            )

        pkl_data = (pdesc, log.getRawData())
        with open(uri, "wb") as f:
            pickle.dump(pkl_data, f)

        return 0

    @hookimpl
    def log(
        self, log: "blocksim.loggers.Logger.Logger", name: str, val: float, tindex: int
    ) -> int:
        return 1
