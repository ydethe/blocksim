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
    def test_suitable(self, logger: Logger) -> bool:
        fic = logger.getLoadedFile()
        if fic is None:
            return False

        istat = fic.endswith(".csv")
        return istat

    @if_suitable
    @hookimpl
    def loadLogFile(self, logger: Logger):
        fic = logger.getLoadedFile()
        data = pd.read_csv(fic, sep=";", na_rep="")
        logger.setRawData(data)

    @if_suitable
    @hookimpl
    def getRawValue(self, logger: Logger, name: str) -> "array":
        """Loads the content of an existing log file

        Args:
          fic
            Path of a log file

        """
        fic = logger.getLoadedFile()
        if fic is None:
            return

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
        df.to_csv(fic, sep=";", na_rep="", index=False)
