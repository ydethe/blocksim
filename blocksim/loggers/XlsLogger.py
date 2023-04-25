import os


import pluggy
import pandas as pd
from singleton3 import Singleton
import numpy as np

from ..utils import FloatArr


__all__ = ["Logger"]

hookimpl = pluggy.HookimplMarker("blocksim")


class Logger(object, metaclass=Singleton):
    @hookimpl
    def test_suitable(self, uri: str) -> bool:
        if uri is None:
            return False

        istat = uri.endswith(".xls")
        return istat

    @hookimpl
    def loadLogFile(self, log: "Logger", uri: str):
        if not self.test_suitable(uri):
            return False

        if not os.path.exists(uri):
            raise FileNotFoundError(uri)

        log.reset()
        data = pd.read_excel(uri, engine="openpyxl", na_values="")
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
    def getRawValue(self, log: "Logger", name: str) -> FloatArr:
        return

    @hookimpl
    def export(self, log: "Logger", uri: str) -> int:
        if not self.test_suitable(uri):
            return -1

        data = log.getRawData()
        df = pd.DataFrame(data)
        df.to_excel(uri, engine="openpyxl", na_rep="", index=False)
        return 0

    @hookimpl
    def log(self, log: "Logger", name: str, val: float, tindex: int) -> int:
        return 1
