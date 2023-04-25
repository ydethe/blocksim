from scipy import linalg as lin
import numpy as np

from blocksim.utils import mean_variance_2dgauss_norm, mean_variance_3dgauss_norm


from blocksim.testing import TestBase


class TestUtils(TestBase):
    def test_mean_variance_2dgauss_norm(self):
        for d in (np.random.uniform(0, 10, size=2) for _ in range(50)):
            d.sort()
            cov = np.diag(d)

            actm, actv = mean_variance_2dgauss_norm(cov)

            ns = 1000000
            X = np.random.multivariate_normal(mean=np.zeros(2), cov=cov, size=ns)
            sn = lin.norm(X, axis=1)

            refm = np.mean(sn)
            refv = np.var(sn)

            self.assertAlmostEqual(actm, refm, delta=5e-3)
            self.assertAlmostEqual(actv, refv, delta=1e-2)

    def test_mean_variance_3dgauss_norm(self):
        for d in (np.random.uniform(0, 10, size=3) for _ in range(50)):
            d.sort()
            cov = np.diag(d)

            actm, actv = mean_variance_3dgauss_norm(cov)

            ns = 1000000
            X = np.random.multivariate_normal(mean=np.zeros(3), cov=cov, size=ns)
            sn = lin.norm(X, axis=1)

            refm = np.mean(sn)
            refv = np.var(sn)

            self.assertAlmostEqual(actm, refm, delta=5e-3)
            self.assertAlmostEqual(actv, refv, delta=1e-2)


if __name__ == "__main__":
    a = TestUtils()
    # a.setUp()
    # a.test_mean_variance_2dgauss_norm()

    a.setUp()
    a.test_mean_variance_3dgauss_norm()
