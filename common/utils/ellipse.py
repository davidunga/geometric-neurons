import numpy as np
import matplotlib.pyplot as plt


def fit_ellipse(x, y):
    """ Based on Halir and Flusser, 'Numerically stable direct least squares fitting of ellipses' """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    coeffs = np.concatenate((ak, T @ ak)).ravel()
    return coeffs


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity
    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r
    e = np.sqrt(1 - r)

    # angle of ccw rotation of the major-axis from x-axis
    if b == 0:
        phi = 0. if a < c else np.pi / 2
    else:
        phi = np.arctan((2. * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2

    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi / 2

    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y



from common.utils.linalg import planar
def get_parabola_pts(coeffs, npts=100, tmin=-2, tmax=2):
    x = np.linspace(tmin, tmax, npts)
    params = cart_to_pol(coeffs)
    x0, y0, ap, bp, e, phi = params
    # vx = x0 - ap * np.cos(phi)
    # vy = y0 - ap * np.sin(phi)
    ang = phi * 180 / np.pi - 90

    # A, B, C, D, E, F = rotate_coeffs(coeffs, np.pi/4 + phi)
    # a = -A / E
    a = .5 * ap / (bp ** 2)
    print("a=",a)
    y = a * x ** 2
    x, y = planar.transform(np.c_[x, y], ang=ang, offset=[vx, vy]).T
    return x, y

if __name__ == '__main__':


    # Test the algorithm with an example elliptical arc.
    npts = 250
    tmin, tmax = np.pi/6, 4 * np.pi/3
    x0, y0 = 4, -3.5
    ap, bp = 7, 3
    ang = 45
    phi = np.radians(np.pi / 4)
    # Get some points on the ellipse (no need to specify the eccentricity).
    x, y = get_ellipse_pts((x0, y0, ap, bp, None, phi), npts, tmin, tmax)

    tmin, tmax = -2, 3
    x = np.linspace(tmin, tmax, npts)
    y = 3 * x ** 2
    x, y = planar.transform(np.c_[x, y], ang=ang, offset=[x0, y0]).T

    noise = 0.001
    x += noise * np.random.normal(size=npts)
    y += noise * np.random.normal(size=npts)

    coeffs = fit_ellipse(x, y)
    print('Exact parameters:')
    print('x0, y0, ap, bp, ang =', x0, y0, ap, bp, ang)
    print('Fitted parameters:')
    print('a, b, c, d, e, f =', coeffs)
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)

    vx = x0 - ap * np.cos(phi)
    vy = y0 - ap * np.sin(phi)
    print('x0, y0, ap, bp, e, phi = ', vx, vy, ap, bp, e, np.degrees(phi))

    plt.plot(x, y, 'x')     # given points
    x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi))
    x, y = get_parabola_pts(coeffs)
    plt.plot(x, y)
    #plt.plot(vx, vy, 'r*')
    plt.show()