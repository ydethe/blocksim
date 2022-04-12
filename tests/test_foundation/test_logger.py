from datetime import datetime
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
    def test_save_load_pickle(self):
        log = Logger()

        pth = Path(__file__).parent / "test_pkl.pkl"

        dt = 0.01
        f = 11
        ns = 1000

        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
            log.log("_", 0)  # Variable named '_' is not recorded

        log.reset()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)

        log.export(pth)
        del log

        log2 = Logger()
        log2.loadLogFile(str(pth))

        vars = log2.getParametersName()
        self.assertIn("t", vars)
        self.assertIn("x", vars)

        tps = np.arange(ns) * dt
        x = exp(1j * tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_save_load_csv(self):
        log = Logger()

        t0 = datetime.now()
        log.setStartTime(t0)
        t1 = log.getStartTime()
        self.assertEqual(t0, t1)

        pth = Path(__file__).parent / "test_ascii.csv"

        dt = 0.01
        f = 11
        ns = 1000

        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
            log.log("_", 0)  # Variable named '_' is not recorded

        log.reset()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)

        log.export(pth)
        del log

        log2 = Logger()
        log2.loadLogFile(pth)

        vars = log2.getParametersName()
        self.assertIn("t", vars)
        self.assertIn("x", vars)

        tps = np.arange(ns) * dt
        x = exp(1j * tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_save_load_xls(self):
        log = Logger()

        pth = Path(__file__).parent / "test_excel.xls"

        dt = 0.01
        f = 11
        ns = 1000

        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
            log.log("_", 0)  # Variable named '_' is not recorded

        log.reset()
        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)

        log.export(pth)

        self.assertRaises(SystemError, log.export, "poney.unknown")

        del log

        log2 = Logger()

        self.assertRaises(SystemError, log2.getRawValue, "poney")

        log2.loadLogFile(pth)

        self.assertRaises(SystemError, log2.getRawValue, "poney")

        vars = log2.getParametersName()
        self.assertIn("t", vars)
        self.assertIn("x", vars)

        tps = np.arange(ns) * dt
        x = exp(1j * tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getRawValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

    def test_log_formatter(self):
        logger.debug("poney")

    def ntest_save_load_psql(self):
        pth = "postgresql+psycopg2://postgres@localhost/simulations"

        log = Logger()

        dt = 0.01
        f = 11
        ns = 1000

        for i in range(ns):
            log.log("x", exp(1j * i * dt * f * 2 * np.pi + 1))
            log.log("t", i * dt)
            log.log("_", 0)  # Variable named '_' is not recorded

        sim_id = log.export(pth)
        del log

        log2 = Logger()
        log2.loadLogFile("%s?sim_id=%i" % (pth, sim_id))

        vars = log2.getParametersName()
        self.assertIn("t", vars)
        self.assertIn("x", vars)

        tps = np.arange(ns) * dt
        x = exp(1j * tps * f * 2 * np.pi + 1)
        err_t = np.max(np.abs(tps - log2.getValue("t")))
        err_x = np.max(np.abs(x - log2.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)


if __name__ == "__main__":
    # unittest.main()

    a = TestLogger()
    a.setUp()
    a.test_save_load_csv()

    # a.setUp()
    # a.test_save_load_xls()

    # a.setUp()
    # a.test_save_load_pickle()
