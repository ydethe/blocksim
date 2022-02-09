from functools import wraps

import pluggy

hookspec = pluggy.HookspecMarker("blocksim")


class LoggerSpec(object):
    @hookspec
    def test_suitable(self, logger) -> bool:
        """Tests weeather a logger can handle a fic"""

    @hookspec
    def loadLogFile(self, logger):
        """Load log file"""

    @hookspec
    def getRawValue(self, logger, name: str) -> "array":
        """Tries to open a stream for writing"""

    @hookspec
    def export(self, logger):
        """Export the log into a file"""


def if_suitable(method):
    @wraps(method)
    def decorated_method(self, logger: "Logger", **kwargs):
        if not self.test_suitable(logger):
            return

        return method(self, logger, **kwargs)

    return decorated_method
