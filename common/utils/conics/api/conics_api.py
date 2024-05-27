

class conic_api:

    @staticmethod
    def kind(): raise NotImplementedError()

    @staticmethod
    def coefficients(m, center, theta): raise NotImplementedError()

    @staticmethod
    def parameters(coeffs): raise NotImplementedError()

    @staticmethod
    def semi_latus(m): raise NotImplementedError()

    @staticmethod
    def eccentricity(m): raise NotImplementedError()

    @staticmethod
    def approx_dist(m, center, theta, pts): raise NotImplementedError()

    @staticmethod
    def nearest_parameter(m, center, theta, pts): raise NotImplementedError()

    @staticmethod
    def nearest_contour_pt(m, center, theta, pts): raise NotImplementedError()

    @staticmethod
    def parametric_pts(m, center, theta, t): raise NotImplementedError()

    @staticmethod
    def focus(m, center, theta): raise NotImplementedError()

    @staticmethod
    def curvature_at_vertex(m): raise NotImplementedError()

    @staticmethod
    def draw(m, center, theta, t=None, *args, **kwargs): raise NotImplementedError()

    @staticmethod
    def kind_name(): raise NotImplementedError

    @staticmethod
    def str(m, center, theta): raise NotImplementedError



