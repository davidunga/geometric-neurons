import unittest
import numpy.testing
from common.utils.procrustes import Procrustes
from common.utils.testing_tools import shapesbank, random_planar_mtx
from common.utils.linalg import planar


class TestProcrustes(unittest.TestCase):

    def setUp(self):
        pass

    def test_affine(self):
        Y = shapesbank.parabola()
        for seed in range(50):
            A_ = random_planar_mtx(seed=seed, dim=3, ortho=False)
            X = planar.apply(A_, Y)
            proc = Procrustes(kind='affine')
            d, A = proc(X, Y)
            numpy.testing.assert_array_almost_equal(A.flatten(), A_.flatten())
            self.assertAlmostEqual(d, 0)

    def test_ortho(self):
        Y = shapesbank.parabola()
        for seed in range(50):
            A_ = random_planar_mtx(seed=seed, dim=3, ortho=True)
            X = planar.apply(A_, Y)
            proc = Procrustes(kind='ortho')
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
