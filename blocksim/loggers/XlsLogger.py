from typing import Iterable
from datetime import datetime

import pluggy
import pandas as pd

from ..LoggerSpec import if_suitable
from ..Logger import Logger
from .. import logger
from ..exceptions import *


__all__ = ["Logger"]

hookimpl = pluggy.HookimplMarker("blocksim")


class Logger(object):
    @hookimpl
    def test_suitable(self, logger) -> bool:
        fic = logger.getLoadedFile()
        if fic is None:
            return False

        istat = fic.endswith(".xls")
        return istat

    @if_suitable
    @hookimpl
    def loadLogFile(self, logger):
        fic = logger.getLoadedFile()
        logger._data = pd.read_excel(fic, engine="openpyxl", na_rep="")

    @if_suitable
    @hookimpl
    def getRawValue(self, logger, name: str) -> "array":
        """Loads the content of an existing log file

        Args:
          fic
            Path of a log file

        """
        lnames = logger.getParametersName()
        if len(lnames) == 0:
            raise SystemError("Logger empty")
        if not name in lnames:
            raise SystemError("Logger has no variable '%s'" % name)

        data = logger.getRawData()
        value = np.array(data[name])

        return value

    @if_suitable
    @hookimpl
    def export(self, logger: Logger):
        fic = logger.getLoadedFile()
        data = logger.getRawData()
        df = pd.DataFrame(data)
        df.to_excel(fic, engine="openpyxl", na_rep="", index=False)
