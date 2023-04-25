# -*- coding: utf-8 -*-
"""

.. include:: ../README.md

# CLI usage

blocksim comes with some CLI utilities.

    blocksim --help

    Usage: blocksim [OPTIONS] COMMAND [ARGS]...

    Options:
    ...
    --help                          Show this message and exit.

    Commands:
    gnss_sim      GNSS constellation simulation

# Examples

[See examples](../examples/index.html)

# Testing blocksim

## Run the tests

To run tests, just run:

    pdm test

## Baseline images generation

If needed (for example, a new test with its associated baseline image), we might have to
regenerate the baseline images. In this case, run:

    pdm baseline

## Test reports

[See test report](../tests/report.html)

[See test results](../tests/results/fig_comparison.html)

[See coverage](../coverage/index.html)

# Class diagram

![classes](./classes.png "Class diagram")

.. include:: ../CHANGELOG.md

"""

# __init__.py
import os
import logging
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

import pluggy
import control
import numpy
from rich.logging import RichHandler

if TYPE_CHECKING:
    from .loggers.Logger import Logger
else:
    Logger = "blocksim.loggers.Logger.Logger"

control.use_numpy_matrix(flag=False, warn=True)


# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger("blocksim_logger")
logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

stream_handler = RichHandler()
logger.addHandler(stream_handler)

numpy.seterr(all="raise")

hookspec = pluggy.HookspecMarker("blocksim")


class LoggerSpec(object):
    """Specification for all plugins that extend the `blocksim.loggers.Logger.Logger` capacities"""

    @hookspec
    def test_suitable(self, uri: str) -> bool:
        """Tests weeather a log:'Logger' can handle a fic

        Args:
            uri: The path or URI indicated by the user

        Returns:
            The result of the test

        """

    @hookspec
    def loadLogFile(self, log: Logger, uri: str) -> bool:
        """Load log file

        Args:
            log: The `blocksim.loggers.Logger.Logger` that contains the data
            uri: The path or URI where the data will be written

        Return:
            True if the file could be loaded

        """

    @hookspec
    def getRawValue(self, log: Logger, name: str):
        """If implemented, subsequent calls to `blocksim.loggers.Logger.Logger.getRawValue`
        will use this method. Otherwise, the internal dictionary of
        `blocksim.loggers.Logger.Logger` is used.

        Args:
            log: The `blocksim.loggers.Logger.Logger` that contains the data
            name: Name of the variable to read

        Return:
            The array of values for the variable *name*

        """

    @hookspec
    def export(self, log: Logger, uri: str) -> int:
        """Export the log into a file

        Args:
            log: The `blocksim.loggers.Logger.Logger` that contains the data
            uri: The path or URI to write in

        Return:
            A positive or null value if the export was successful

        """

    @hookspec
    def log(self, log: Logger, name: str, val: float, tindex: int) -> int:
        """Export the log into a file

        Args:
            log: The `blocksim.loggers.Logger.Logger` that contains the data
            name: Name of the parameter. Nothing is logged if *name* == '_'
            val: Value to log
            tindex: Index where the value should be written in the inner data dictionary.
                If Logger.allocate has not been called before, raises an error

        Return:
            A positive or null value if the export was successful

        """


plugin_manager = pluggy.PluginManager("blocksim")
plugin_manager.add_hookspecs(LoggerSpec)

eps = entry_points()
plugins = eps.select(group="blocksim")

for ep in plugins:
    if ep.name.startswith("logger_"):
        try:
            plugin = import_module(ep.value)
        except BaseException as e:
            plugin = None
            logger.warning(f"Failed to load {ep.value}: {e}")
        if plugin is None:
            continue
        if not plugin_manager.is_registered(plugin=plugin.Logger()):
            plugin_manager.register(plugin=plugin.Logger(), name=ep.name)
            logger.info("Registered %s" % ep.value)
