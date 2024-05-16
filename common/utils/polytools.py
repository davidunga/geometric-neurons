import numpy as np
from scipy.interpolate import interp1d
from common.utils.typings import *
from PIL import Image, ImageDraw


def edge_lengths(X: NpPoints):
    return np.linalg.norm(np.diff(X, axis=0), axis=1)


def cumm_arclen(X: NpPoints) -> NpVec[float]:
    return np.r_[0, np.cumsum(edge_lengths(X))]


def total_arclen(X: NpPoints) -> float:
    return cumm_arclen(X)[-1]


def uniform_resample(X: NpPoints, n: int, kind: str = 'linear') -> NpPoints:
    s = cumm_arclen(X)
    s_new = np.linspace(0, s[-1], n)
    X_new = interp1d(x=s, y=X.T, kind=kind)(s_new).T
    return X_new


def rasterize_paths(paths: Sequence[np.ndarray], image_wh, width: int = 1, reduce: str = 'mean', **kwargs):

    # -------
    span_inlier_pcntl = 1
    margin = .05
    # -------

    image_wh = np.asarray(image_wh)
    ww, hh = (1 - 2 * margin) * image_wh
    all_paths = np.vstack(paths)
    xspan = np.diff(np.percentile(all_paths[:, 0], [span_inlier_pcntl, 100 - span_inlier_pcntl]))[0]
    yspan = np.diff(np.percentile(all_paths[:, 0], [span_inlier_pcntl, 100 - span_inlier_pcntl]))[0]
    scale = np.array([ww / xspan, hh / yspan])
    offset = image_wh / 2 - np.median(all_paths, keepdims=True)
    images = np.zeros((len(paths), *image_wh))
    for i, pts in enumerate(paths):
        image = Image.new('L', (image_wh[0], image_wh[1]), 0)
        draw = ImageDraw.Draw(image)
        draw.line([(x, y) for (x, y) in (pts * scale + offset).round().astype(int)], fill=1, width=width, **kwargs)
        images[i] = np.array(image)

    if reduce == 'none':
        ret = images
    else:
        ret = getattr(np, reduce)(images, axis=0)

    return ret, scale, offset.squeeze()


def _test():
    from common.utils.testing_tools import shapesbank
    a = 1
    b = 1.5
    X = shapesbank.ellipse(a=a, b=b, n=50000)
    approx_true_arclen = (a + b) * np.pi
    print("approx true arclen =", approx_true_arclen)
    print("est total arclen=", total_arclen(X))

if __name__ == "__main__":
    _test()