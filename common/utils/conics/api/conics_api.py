

class conic_api:

    @staticmethod
    def kind(): raise NotImplementedError()

    @staticmethod
    def coefficients(m, loc, ang): raise NotImplementedError()

    @staticmethod
    def parameters(coeffs): raise NotImplementedError()

    @staticmethod
    def semi_latus(m): raise NotImplementedError()

    @staticmethod
    def eccentricity(m): raise NotImplementedError()

    @staticmethod
    def radii_ratio_from_eccentricity(e: float): raise NotImplementedError()

    @staticmethod
    def approx_dist(m, loc, ang, pts): raise NotImplementedError()

    @staticmethod
    def nearest_p(m, loc, ang, pts): raise NotImplementedError()

    @staticmethod
    def nearest_t(m, loc, ang, pts): raise NotImplementedError()

    @staticmethod
    def t_to_p(m, t): raise NotImplementedError()

    @staticmethod
    def p_to_t(m, p): raise NotImplementedError()

    @staticmethod
    def nearest_contour_pt(m, loc, ang, pts): raise NotImplementedError()

    @staticmethod
    def parametric_pts(m, loc, ang, p, ptype: str = 'p'): raise NotImplementedError()

    @staticmethod
    def focus_pts(m, loc, ang): raise NotImplementedError()

    @staticmethod
    def vertex_pts(m, loc, ang): raise NotImplementedError()

    @staticmethod
    def curvature_at_vertex(m): raise NotImplementedError()

    @staticmethod
    def draw(m, loc, ang, t=None, *args, **kwargs): raise NotImplementedError()

    @staticmethod
    def kind_name(): raise NotImplementedError

    @staticmethod
    def str(m, loc, ang): raise NotImplementedError

    @staticmethod
    def focus_to_vertex_dist(m): raise NotImplementedError
