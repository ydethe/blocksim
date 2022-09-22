from typing import Any
import os

from nptyping import NDArray
import pluggy
import pandas as pd
from singleton3 import Singleton

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

        istat = uri.endswith(".csv")
        return istat

    @hookimpl
    def loadLogFile(self, log: "blocksim.loggers.Logger.Logger", uri: str) -> bool:
        if not self.test_suitable(uri):
            return False

        if not os.path.exists(uri):
            raise FileNotFoundError(uri)

        log.reset()
        data = pd.read_csv(uri, sep=";", na_values="")
        for k in data.columns:
            unit = ""
            typ = data[k].dtype
            if data[k].dtype == "O":
                data[k] = data[k].apply(np.complex128)
                typ = np.complex128
            log.createEmptyValue(name=k, unit=unit, description="", dtype=typ)
        log.setRawData(data)
        return True

    @hookimpl
    def getRawValue(self, log: "blocksim.loggers.Logger.Logger", name: str) -> NDArray[Any, Any]:
        return

    @hookimpl
    def export(self, log: "blocksim.loggers.Logger.Logger", uri: str) -> int:
        if not self.test_suitable(uri):
            return -1

        data = log.getRawData()
        df = pd.DataFrame(data)
        df.to_csv(uri, sep=";", na_rep="", index=False)
        return 0

    @hookimpl
    def log(self, log: "blocksim.loggers.Logger.Logger", name: str, val: float, tindex: int) -> int:
        return 1
