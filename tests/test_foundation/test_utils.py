from scipy import linalg as lin
import numpy as np
from numpy import pi
from hypothesis import given, settings, strategies as st

from blocksim.utils import (
    mean_variance_2dgauss_norm,
    mean_variance_3dgauss_norm,
    is_above_elevation_mask,
    itrf_to_azeld,
    llavpa_to_itrf,
    rotation_matrix,
    compute_axis_angle,
)
from blocksim.testing import TestBase, st_lon, st_lat, st_elev, st_lonlatalt


class TestUtils(TestBase):
    @given(
        axis=st.tuples(
            st.sampled_from(np.linspace(-1, 1, 201)),
            st.sampled_from(np.linspace(-1, 1, 201)),
            st.sampled_from(np.linspace(-1, 1, 201)),
        ),
        angle=st.sampled_from(np.linspace(-2 * pi, 2 * pi, 201)),
    )
    @settings(print_blob=True)
    def test_rotation_matrix(self, axis, angle):
        np_axis = np.array(axis)
        if lin.norm(axis) < 1e-6:
            return

        R = rotation_matrix(angle, np_axis)
        axis2, angle2 = compute_axis_angle(R)
        if axis2 @ np_axis < 0:
            axis2 *= -1
            angle2 *= -1
        angle_err = angle2 - angle
        while angle_err > pi:
            angle_err -= 2 * pi
        while angle_err < -pi:
            angle_err += 2 * pi

        self.assertAlmostEqual(angle_err, 0)

        if lin.norm(angle % (2 * pi)) > 1e-3:
            err = lin.norm(np.cross(np_axis, axis2))
            self.assertAlmostEqual(err, 0)

    @given(
        obs_ll=st.tuples(st_lon, st_lat),
        sat_lla=st_lonlatalt,
        mask=st_elev,
    )
    @settings(print_blob=True)
    def test_elevation_mask(self, obs_ll, sat_lla, mask: float):
        obs = llavpa_to_itrf((obs_ll[0], obs_ll[1], 0, 0, 0, 0))
        sat = llavpa_to_itrf((sat_lla[0], sat_lla[1], sat_lla[2], 0, 0, 0))

        _, el, _, _, _, _ = itrf_to_azeld(obs, sat)
        ref = el > mask
        assert ref == is_above_elevation_mask(obs, sat, mask)

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

    # a.setUp()

    # a.test_mean_variance_3dgauss_norm()
    # a.test_elevation_mask()
    a.test_rotation_matrix()
