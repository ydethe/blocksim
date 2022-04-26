"""Definition of the spec for a logger plugin

"""

from functools import wraps
from pathlib import Path

import pluggy

hookspec = pluggy.HookspecMarker("blocksim")


class LoggerSpec(object):
    """Specification for all plugins that extend the `blocksim.loggers.Logger.Logger` capacities"""

    @hookspec
    def test_suitable(self, uri: Path) -> bool:
        """Tests weeather a log:'Logger' can handle a fic

        Args:
            uri: The path or URI indicated by the user

        Returns:
            The result of the test

        """

    @hookspec
    def loadLogFile(self, log: "Logger", uri: Path) -> bool:
        """Load log file

        Args:
            log: The `blocksim.loggers.Logger.Logger` that contains the data
            uri: The path or URI where the data will be written

        Return:
            True if the file could be loaded

        """

    @hookspec
    def getRawValue(self, log: "Logger", name: str) -> "array":
        """If implemented, subsequent calls to `blocksim.loggers.Logger.Logger.getRawValue` will use this method.
        Otherwise, the internal dictionary of `blocksim.loggers.Logger.Logger` is used.

        Args:
            log: The `blocksim.loggers.Logger.Logger` that contains the data
            name: Name of the variable to read

        Return:
            The array of values for the variable *name*

        """

    @hookspec
    def export(self, log: "Logger", uri: Path) -> int:
        """Export the log into a file

        Args:
            log: The `blocksim.loggers.Logger.Logger` that contains the data
            uri: The path or URI to write in

        Return:
            A positive or null value if the export was successful


        """
