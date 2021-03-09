import os
import unittest
from typing import Iterable
import pathlib
from inspect import currentframe, getframeinfo

import numpy as np


class TestBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1883647)
