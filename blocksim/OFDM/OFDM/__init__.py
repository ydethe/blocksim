from pkg_resources import get_distribution
import logging
import os

from OFDM.LogFormatter import LogFormatter


try:
    __version__ = get_distribution(__name__).version
except Exception as e:
    __version__ = "dev"

__author__ = "Y. BLAUDIN DE THE"
__email__ = "yann.blaudin-de-the@thalesaleniaspace.com"


logger = logging.getLogger("ofdm_logger")

formatter = LogFormatter()
file_handler = logging.FileHandler("ofdm.log", mode="w", encoding="utf-8", delay=False)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

formatter = LogFormatter()
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

logger.setLevel(logging.DEBUG)
