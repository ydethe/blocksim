# __init__.py
from pkg_resources import get_distribution
import logging
from datetime import datetime
import os

from .LogFormatter import LogFormatter

try:
    import control
    control.use_numpy_matrix(flag=False, warn=True)
except:
    pass

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
