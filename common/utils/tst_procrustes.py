import numpy as np
from common.utils.procrustes import Procrustes
from common.utils.testing_tools import shapesbank, random_planar_mtx, random_planar_mtx_by_kind
from common.utils.linalg import planar
import matplotlib.pyplot as plt

# Didn't use unittest because it suppresses print()
# Using 'tst' instead of 'test' so pyCharm won't run it as unittest


class UnitTst:

    def __init__(self, atol: float = 1e-6, rtol: float = 1e-5, n_iters: int = 50):
        self.rtol = rtol
        self.atol = atol
        self.n_iters = n_iters

    def assert_almost_equal(self, a, b):
        assert abs(a - b) < self.atol

    def assert_not_almost_equal(self, a, b):
        assert abs(a - b) >= self.atol

    def run(self):
        test_func_names = [member for member in dir(self)
                           if member.startswith('test') and callable(getattr(self, member))]
        for func_name in test_func_names:
            print(f"{func_name}:", end="")
            func = getattr(self, func_name)
            try:
                func()
                print(" PASSED")
            except:
                raise


class TstProcrustes(UnitTst):

    def _test(self, kind: str):
        # Test that Procrustes correctly reconstructs transformation
        Y = shapesbank.parabola()
        procrustes = Procrustes(kind=kind)
        for seed in range(self.n_iters):
            A_true = random_planar_mtx_by_kind(seed=seed, kind=kind)
            X = planar.apply(A_true, Y)
            d, A_hat, _ = procrustes(X, Y)
            procrustes.verify_matrix_kind(A_hat)
            np.testing.assert_array_almost_equal(A_true.flatten(), A_hat.flatten())
            self.assert_almost_equal(d, 0)

    def test_affine(self):
        self._test(kind='affine')

    def test_ortho(self):
        self._test(kind='ortho')

    def test_rigid(self):
        self._test(kind='rigid')

    def test_error(self):
        # test that error behaves as expected

        X1 = shapesbank.parabola(t0=-2, t1=1, n=50)

        assert 0 <= Procrustes.error(X1, X1) <= 1e-16

        # check that error grows with relative shape scale and offset:
        values = np.linspace(1, 10, 50)
        assert np.all(np.diff([Procrustes.error(X1, X1 * v) for v in values]) > 0)
        assert np.all(np.diff([Procrustes.error(X1, X1 / v) for v in values]) > 0)
        assert np.all(np.diff([Procrustes.error(X1, X1 + v) for v in values]) > 0)
        assert np.all(np.diff([Procrustes.error(X1, X1 - v) for v in values]) > 0)

        # check that applying the same transform to both shapes does not change error:
        for factor in np.linspace(1.01, 10, 50):
            X2 = factor * X1
            errors = []
            for seed in range(self.n_iters):
                A = random_planar_mtx_by_kind(seed=seed, kind='ortho')
                rX1 = planar.apply(A, X1)
                rX2 = planar.apply(A, X2)
                errors.append(Procrustes.error(rX1, rX2))
            assert np.std(errors) < self.rtol * np.mean(errors)


def show_error_behaviour():
    n = 1000
    mx = 500
    seed = 1

    offsets = np.linspace(0, mx, n)
    scales = np.linspace(1, mx, n)
    rng = np.random.default_rng(seed)

    shapes = {
        'parabola1': shapesbank.parabola(t0=-2, t1=1, n=50),
        'parabola2': shapesbank.parabola(t0=-5, t1=2, n=500),
        'ellipse1': shapesbank.ellipse(a=10, b=5, n=50),
    }

    _, axs = plt.subplots(nrows=len(shapes))

    for shape, ax in zip(shapes, axs):

        plt.sca(ax)
        X1 = shapes[shape]
        X1 = planar.apply(random_planar_mtx_by_kind(kind='ortho', seed=1), X1)
        sigmas = np.std(X1, axis=0).max() * scales

        errors = [Procrustes.error(X1, X1 + offset) for offset in offsets]
        plt.plot(offsets, errors, '-', label='Offset')

        errors = [Procrustes.error(X1, X1 * scale) for scale in scales]
        plt.plot(offsets, errors, '-', label='Scale')

        errors = [Procrustes.error(X1, X1 + sigma * rng.standard_normal(size=X1.shape)) for sigma in sigmas]
        plt.plot(scales, errors, '-', label='Noise')

        plt.title(shape)

    plt.ylabel("Error")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    show_error_behaviour()
    #TstProcrustes().run()

