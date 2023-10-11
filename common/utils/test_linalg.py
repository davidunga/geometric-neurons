import unittest
import numpy as np
from common.utils.linalg import planar, lsqr


class TestTform(unittest.TestCase):

    def test_check_mtx(self):
        # Valid matrices
        A1 = np.eye(2)
        A2 = np.eye(3)
        planar._check_mtx(A1)
        planar._check_mtx(A2)

        # Invalid matrices
        A3 = np.array([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(AssertionError):
            planar._check_mtx(A3)

    def test_build(self):
        A = planar.build(b=2.0, ang=45.0, t=[1.0, 1.0], reflect='x')
        self.assertEqual(A.shape, (3, 3))

    def test_apply_and_apply_inv(self):
        A = planar.build()
        X = np.array([[1, 2], [3, 4]])
        Y = planar.apply(A, X)
        X_inv = planar.apply_inv(A, Y)
        np.testing.assert_array_almost_equal(X, X_inv)

    def test_decompose(self):

        rng = np.random.default_rng(1)

        for _ in range(50):

            b_ = rng.random() * 5
            reflect_ = rng.choice(['none', 'x', 'y'])
            ang_ = rng.random() * (90 if reflect_ != 'none' else 360)
            ang_ = min(ang_, 360 - ang_)
            t_ = 10 * rng.standard_normal(2)

            A = planar.build(b=b_, ang=ang_, t=t_, reflect=reflect_)

            scale_metric = rng.choice(['det', 'avg'])
            b, ang, t, is_reflected, ortho_score = planar.decompose(A, scale_metric=scale_metric)
            ang = min(ang, 360 - ang)

            self.assertAlmostEqual(ortho_score, 1)
            self.assertTrue(isinstance(b, float))
            self.assertTrue(isinstance(ang, float))
            self.assertTrue(isinstance(t, np.ndarray))
            self.assertTrue(isinstance(is_reflected, (bool, np.bool_)))
            self.assertAlmostEqual(b, b_)
            self.assertAlmostEqual(ang, ang_)
            self.assertListEqual(list(t), list(t_))
            self.assertEqual(reflect_ != 'none', is_reflected)

    def test_linlsqr(self):
        x = np.array([[1, 2], [3, 4]])
        b = np.array([[2, 3], [4, 5]])
        A = lsqr(x, b)
        self.assertEqual(A.shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
