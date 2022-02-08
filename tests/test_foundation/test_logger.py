import sys
import os
from pathlib import Path
import unittest

import numpy as np
from numpy import pi, exp, sin
from matplotlib import pyplot as plt
import pytest

from blocksim import logger
from blocksim.Logger import Logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestLogger(TestBase):
    def test_save_load_psql(self):
        pth = "postgresql+psycopg2://%s:%s@postgres/%s" % (
            os.environ["POSTGRES_USER"],
            os.environ["POSTGRES_PASSWORD"],
            os.environ["POSTGRES_DB"],
        )

        log = Logger()
        log.setOutputLoggerFile(fic=pth)

        dt = 0.01
        f = 11
        ns = 1000

        log.openFile()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
            log.log("_", 0)  # Variable named '_' is not recorded

        sim_id = log.closeFile()
        del log

        log2 = Logger()
        log2.loadLoggerFile("%s?sim_id=%i" % (pth, sim_id))

        vars = log2.getParametersName()
        self.assertIn("t", vars)
        self.assertIn("x", vars)

        tps = np.arange(ns) * dt
        x = exp(1j * tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_save_load_parquet(self):
        log = Logger()
        self.assertFalse(log.hasOutputLoggerFile())

        pth = Path(__file__).parent / "test_pkl.pkl"

        self.assertRaises(
            AssertionError, log.setOutputLoggerFile, fic=str(pth), binary=False
        )

        log.setOutputLoggerFile(fic=str(pth), binary=True)
        self.assertTrue(log.hasOutputLoggerFile())

        dt = 0.01
        f = 11
        ns = 1000

        log.openFile()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
            log.log("_", 0)  # Variable named '_' is not recorded

        log.reset()
        log.openFile()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)

        log.closeFile()
        del log

        log2 = Logger()
        log2.loadLoggerFile(str(pth), binary=True)

        vars = log2.getParametersName()
        self.assertIn("t", vars)
        self.assertIn("x", vars)

        tps = np.arange(ns) * dt
        x = exp(1j * tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_save_load_ascii(self):
        log = Logger()
        self.assertFalse(log.hasOutputLoggerFile())

        pth = Path(__file__).parent / "test_ascii.log"

        log.setOutputLoggerFile(str(pth))
        self.assertTrue(log.hasOutputLoggerFile())

        dt = 0.01
        f = 11
        ns = 1000

        log.openFile()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
            log.log("_", 0)  # Variable named '_' is not recorded

        log.reset()
        log.openFile()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)

        del log

        log2 = Logger()
        log2.loadLoggerFile(str(pth))

        vars = log2.getParametersName()
        self.assertIn("t", vars)
        self.assertIn("x", vars)

        tps = np.arange(ns) * dt
        x = exp(1j * tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_load_binary_v1(self):
        pth = Path(__file__).parent / "test_bin_v1.log"

        log = Logger()
        log.loadLoggerFile(str(pth), binary=True)

        dt = 0.01
        f = 11
        ns = 1000

        tps = np.arange(ns) * dt
        x = sin(tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log.getValue("t")))
        err_x = np.max(np.abs(x - log.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_save_load_binary(self):
        log = Logger()
        self.assertFalse(log.hasOutputLoggerFile())

        pth = Path(__file__).parent / "test_bin.log"

        log.setOutputLoggerFile(str(pth), binary=True)
        self.assertTrue(log.hasOutputLoggerFile())

        dt = 0.01
        f = 11
        ns = 1000

        log.openFile()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
        del log

        log2 = Logger()
        log2.loadLoggerFile(str(pth), binary=True)

        tps = np.arange(ns) * dt
        x = exp(1j * tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_log_formatter(self):
        logger.debug("poney")


if __name__ == "__main__":
    # unittest.main()

    a = TestLogger()
    # a.setUp()
    # a.test_save_load_ascii()

    # a.setUp()
    # a.test_save_load_binary()

    # a.setUp()
    # a.test_load_binary_v1()

    # a.setUp()
    # a.test_save_load_parquet()

    a.setUp()
    a.ntest_save_load_psql()
