from operator import methodcaller
from typing import Iterable
from datetime import datetime

import pluggy
import pandas as pd
from singleton3 import Singleton

from ..Logger import Logger
from .. import logger
from ..exceptions import *


__all__ = ["Logger"]

hookimpl = pluggy.HookimplMarker("blocksim")


class Logger(object, metaclass=Singleton):
    @hookimpl
    def test_suitable(self, fic: str) -> bool:
        if fic is None:
            return False

        istat = fic.endswith(".pkl")
        return istat

    @hookimpl
    def loadLogFile(self, log: "Logger", fic: str):
        if not self.test_suitable(fic):
            return False

        data = pd.read_pickle(fic)
        log.setRawData(data)
        return True

    @hookimpl
    def getRawValue(self, log: "Logger", name: str) -> "array":
        """Loads the content of an existing log file

        Args:
          fic
            Path of a log file

        """
        return

    @hookimpl
    def export(self, log: "Logger", fic: str) -> int:
        if not self.test_suitable(fic):
            return -1

        data = log.getRawData()
        df = pd.DataFrame(data)
        df.to_pickle(fic)
        return 0
