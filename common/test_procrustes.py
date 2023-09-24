import unittest
import numpy as np
import numpy.testing

from analysis.procrustes import Procrustes
from common.shapesbank import shapesbank
from common.linalg import tform


def random_mtx(seed: int = 0, det: float = 1.0, dim: int = 2):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    A[:2, :2] = (det + A[0, 1] * A[1, 0]) / A[0, 0]
    if dim == 3:
        A[2] = [0, 0, 1]
    return A


class TestProcrustes(unittest.TestCase):

    def setUp(self):
        pass

    def test_affine(self):
        Y = shapesbank.parabola()
        A_ = random_mtx(seed=1, dim=3)
        X = tform.apply(A_, Y)
        proc = Procrustes(kind='affine')
        d, A = proc(X, Y)
        numpy.testing.assert_array_almost_equal(A.flatten(), A_.flatten())
        self.assertAlmostEqual(d, 0)


    def test_error(self):
        X1 = shapesbank.parabola()
        X2 = 1.1 * shapesbank.parabola()
        self.assertAlmostEqual(Procrustes.error(X1, X1), 0)
        self.assertAlmostEqual(Procrustes.error(X2, X2), 0)
        self.assertNotAlmostEqual(Procrustes.error(X1, X2), 0)


if __name__ == '__main__':
    unittest.main()
