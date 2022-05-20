"""

.. include:: ../README.md

# CLI usage

blocksim comes with some CLI utilities.

    blocksim --help

    Usage: blocksim [OPTIONS] COMMAND [ARGS]...

    Options:
    --install-completion [bash|zsh|fish|powershell|pwsh]
                                    Install completion for the specified shell.
    --show-completion [bash|zsh|fish|powershell|pwsh]
                                    Show completion for the specified shell, to
                                    copy it or customize the installation.
    --help                          Show this message and exit.

    Commands:
    db      Manage databases.
    header  Visualize a binary log file's header

# Examples

[See examples](../examples/index.html)

# Testing blocksim

## Run the tests

To run tests, just run:

    python3 -m pytest -n 8 --html=report.html --self-contained-html --mpl --mpl-generate-summary=html --mpl-baseline-path=tests/baseline --mpl-results-path=results --cov blocksim tests --doctest-modules blocksim

To only run the tests related to the unstaged files or the current branch (according to Git):

    python3 -m pytest -n 8 --picked --html=report.html --self-contained-html --mpl --mpl-generate-summary=html --mpl-baseline-path=tests/baseline --mpl-results-path=results --cov blocksim tests --doctest-modules blocksim

## Code coverage

Once the tests are run, the code coverage is available. To have a html version in the htmlcov folder, run:

    coverage html

[See coverage](../coverage/index.html)

## Baseline images generation

If needed (for example, a new test with its associated baseline image), we might have to regenerate the baseline images. In this case, run:

    python3 -m pytest -n 8 --mpl-generate-path=tests/baseline tests

## Test reports

[See test report](../tests/report.html)

[See test results](../tests/results/fig_comparison.html)

# Class diagram

![classes](./classes.png "Class diagram")

# Distribution

Download source archive and wheel file [distribution](../dist)

"""

# __init__.py
import os
from pkg_resources import get_distribution
import logging
from importlib import import_module
from importlib.metadata import entry_points

from pluggy import PluginManager
import control
import numpy

control.use_numpy_matrix(flag=False, warn=True)

from .loggers.LoggerSpec import LoggerSpec
from .LogFormatter import LogFormatter


__version__ = get_distribution(__name__).version

__author__ = "Y. BLAUDIN DE THE"
__email__ = "yann.blaudin-de-the@thalesaleniaspace.com"


# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger("blocksim_logger")

# on met le niveau du logger à DEBUG, comme ça il écrit tout
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
formatter = LogFormatter()
# création d'un handler qui va rediriger chaque écriture de log
# sur la console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

numpy.seterr(all="raise")

plugin_manager = PluginManager("blocksim")
plugin_manager.add_hookspecs(LoggerSpec)

eps = entry_points()
plugins = eps["blocksim"]

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
