from typing import Any
from pathlib import Path

from nptyping import NDArray, Shape
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
    def test_suitable(self, uri: Path) -> bool:
        if uri is None:
            return False

        istat = uri.suffix == ".csv"
        return istat

    @hookimpl
    def loadLogFile(self, log: "Logger", uri: Path) -> bool:
        if not self.test_suitable(uri):
            return False

        if not uri.exists():
            raise FileNotFoundError(uri)

        data = pd.read_csv(uri, sep=";", na_values="")
        for k in data.columns:
            if data[k].dtype == "O":
                data[k] = data[k].apply(np.complex128)
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
        df.to_csv(uri, sep=";", na_rep="", index=False)
        return 0

    @hookimpl
    def log(self, log: "Logger", name: str, val: float) -> int:
        return 1
