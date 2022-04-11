from functools import wraps

import pluggy

hookspec = pluggy.HookspecMarker("blocksim")


class LoggerSpec(object):
    """Specification for all plugins that extend the `Logger.Logger` capacities"""

    @hookspec
    def test_suitable(self, fic: str) -> bool:
        """Tests weeather a log:'Logger' can handle a fic

        Args:
            fic: A path indicated by the user

        Returns:
            The result of the test

        """

    @hookspec
    def loadLogFile(self, log: "Logger", fic: str) -> bool:
        """Load log file

        Args:
            log: The `Logger.Logger` that contains the data
            fic: The fic where the data will be written

        Return:
            True if the file could be loaded

        """

    @hookspec
    def getRawValue(self, log: "Logger", name: str) -> "array":
        """If implemented, subsequent calls to `Logger.Logger.getRawValue` will use this method.
        Otherwise, the internal dictionary of `Logger.Logger` is used.

        Args:
            log: The `Logger.Logger` that contains the data
            name: Name of the variable to read

        Return:
            The array of values for the variable *name*

        """

    @hookspec
    def export(self, log: "Logger", fic: str) -> int:
        """Export the log into a file

        Args:
            log: The `Logger.Logger` that contains the data
            fic: The path to the file to write in

        Return:
            A positive or null value if the export was successful


        """
