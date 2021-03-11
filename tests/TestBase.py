import os
import unittest
from typing import Iterable
import pathlib
from inspect import currentframe, getframeinfo

import numpy as np
from matplotlib import pyplot as plt

from blocksim.Graphics import plotFromLogger


class TestBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1883647)

    def plotVerif(self, var: str, axe, **kwargs):
        axe.grid(True)

        plotFromLogger(self.log, "t", var, axe, **kwargs)
