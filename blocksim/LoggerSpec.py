from functools import wraps

import pluggy

hookspec = pluggy.HookspecMarker("blocksim")


class LoggerSpec(object):
    @hookspec
    def test_suitable(self, fic: str) -> bool:
        """Tests weeather a log:'Logger' can handle a fic"""

    @hookspec
    def loadLogFile(self, log: "Logger") -> bool:
        """Load log file"""

    @hookspec
    def getRawValue(self, log: "Logger", name: str) -> "array":
        """Tries to open a stream for writing"""

    @hookspec
    def export(self, log: "Logger") -> int:
        """Export the log into a file"""
