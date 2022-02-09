# __init__.py
from pkg_resources import get_distribution
import logging
from importlib import import_module
from importlib.metadata import entry_points

from pluggy import PluginManager
import control

control.use_numpy_matrix(flag=False, warn=True)

from .LoggerSpec import LoggerSpec
from .LogFormatter import LogFormatter


__version__ = get_distribution(__name__).version

__author__ = "Y. BLAUDIN DE THE"
__email__ = "ydethe@gmail.com"


# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger("blocksim_logger")

# on met le niveau du logger à DEBUG, comme ça il écrit tout
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
formatter = LogFormatter()
# création d'un handler qui va rediriger chaque écriture de log
# sur la console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

plugin_manager = PluginManager("blocksim")
plugin_manager.add_hookspecs(LoggerSpec)

eps = entry_points()
plugins = eps["blocksim"]

for ep in plugins:
    plugin = import_module(ep.value)
    plugin_manager.register(plugin.Logger())
    logger.info("Registered %s.Logger()" % ep.value)
