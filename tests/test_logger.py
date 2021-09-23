import os
import sys
import unittest

import numpy as np
from matplotlib import pyplot as plt
import pytest

from blocksim.Logger import Logger
from blocksim.Graphics import plotFromLogger


sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestLogger(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_logger(self):
        log = Logger()

        dt = 0.01
        f = 11
        ns = 1000

        for i in range(ns):
            log.log("t", i * dt)
            log.log("x", np.sin(i * dt * f * 2 * np.pi + 1))

        tps = np.arange(ns) * dt
        x = np.sin(tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log.getValue("t")))
        err_x = np.max(np.abs(x - log.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

        fig = plt.figure()
        fig.suptitle = "Essai logger"
        axe = fig.add_subplot(111)
        axe.grid(True)

        fc = 5.0
        fr = log.getFilteredValue("x", 64, 2 * fc * dt)

        plotFromLogger(log, "t", "x", axe, label="brut")
        plotFromLogger(log, "t", fr, axe, label="filtré")
        axe.legend(loc="best")

        return fig

    def test_save_load_ascii(self):
        log = Logger()
        self.assertFalse(log.hasOutputLoggerFile())

        log.setOutputLoggerFile("tests/test_ascii.log")
        self.assertTrue(log.hasOutputLoggerFile())

        dt = 0.01
        f = 11
        ns = 1000

        log.openFile()
        for i in range(ns):
            log.log("x", np.sin(i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
        del log

        log2 = Logger()
        log2.loadLoggerFile("tests/test_ascii.log")

        tps = np.arange(ns) * dt
        x = np.sin(tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_save_load_binary(self):
        log = Logger()
        self.assertFalse(log.hasOutputLoggerFile())

        log.setOutputLoggerFile("tests/test_bin.log", binary=True)
        self.assertTrue(log.hasOutputLoggerFile())

        dt = 0.01
        f = 11
        ns = 1000

        log.openFile()
        for i in range(ns):
            log.log("x", np.sin(i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
        del log

        log2 = Logger()
        log2.loadLoggerFile("tests/test_bin.log", binary=True)

        tps = np.arange(ns) * dt
        x = np.sin(tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)


if __name__ == "__main__":
    # unittest.main()

    a = TestLogger()
    a.test_save_load_binary()
