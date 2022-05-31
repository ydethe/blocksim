from typing import Any
from pathlib import Path

from nptyping import NDArray, Shape
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
    def test_suitable(self, uri: Path) -> bool:
        if uri is None:
            return False

        istat = uri.suffix == ".pkl"
        return istat

    @hookimpl
    def loadLogFile(self, log: "Logger", uri: Path):
        if not self.test_suitable(uri):
            return False

        if not uri.exists():
            raise FileNotFoundError(uri)

        log.reset()
        data = pd.read_pickle(uri)
        for name in data.columns:
            unit = ""
            typ = data[name].dtype
            log.createEmptyValue(name=name, unit=unit, description="", dtype=typ)
        log.setRawData(data)
        return True

    @hookimpl
    def getRawValue(self, log: "Logger", name: str) -> NDArray[Any, Any]:
        return

    @hookimpl
    def export(self, log: "Logger", uri: Path) -> int:
        if not self.test_suitable(uri):
            return -1

        data = log.getRawData()
        df = pd.DataFrame(data)
        df.to_pickle(uri)
        return 0

    @hookimpl
    def log(self, log: "Logger", name: str, val: float, tindex: int) -> int:
        return 1
