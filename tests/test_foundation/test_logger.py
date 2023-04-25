from datetime import datetime
from pathlib import Path

import numpy as np
from numpy import exp

from blocksim import logger
from blocksim.loggers.Logger import Logger


from blocksim.testing import TestBase


class TestLogger(TestBase):
    def test_save_load_pickle(self):
        log = Logger()

        pth = Path(__file__).parent / "test_pkl.pkl"

        dt = 0.01
        f = 11
        ns = 1000

        for i in range(ns):
            log.log(name="x", val=exp(1j * i * dt * f * 2 * np.pi + 1), unit="")
            log.log(name="t", val=i * dt, unit="t")
            log.log(name="_", val=0, unit="")  # Variable named '_' is not recorded

        log.reset()
        for i in range(ns):
            log.log(name="x", val=exp(1j * i * dt * f * 2 * np.pi + 1), unit="")
            log.log(name="t", val=i * dt, unit="s")

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
            log.log(name="x", val=exp(1j * i * dt * f * 2 * np.pi + 1), unit="")
            log.log(name="t", val=i * dt, unit="t")
            log.log(name="_", val=0, unit="")  # Variable named '_' is not recorded

        log.reset()
        for i in range(ns):
            log.log(name="x", val=exp(1j * i * dt * f * 2 * np.pi + 1), unit="")
            log.log(name="t", val=i * dt, unit="t")

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
            log.log(name="x", val=exp(1j * i * dt * f * 2 * np.pi + 1), unit="")
            log.log(name="t", val=i * dt, unit="s")
            log.log(name="_", val=0, unit="")  # Variable named '_' is not recorded

        log.reset()
        for i in range(ns):
            log.log(name="x", val=exp(1j * i * dt * f * 2 * np.pi + 1), unit="")
            log.log(name="t", val=i * dt, unit="s")

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


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    a = TestLogger()
    # a.setUp()
    # a.test_save_load_parquet()

    # a.setUp()
    # a.test_save_load_csv()

    a.setUp()
    a.test_save_load_xls()

    # a.setUp()
    # a.test_save_load_pickle()
