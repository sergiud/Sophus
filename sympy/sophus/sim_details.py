import unittest

import sympy


class SimDetails:
    """ Symbolic derivations backing the numerically stable coefficient
        formulas used by sophus/sim_details.hpp for the Sim(N) exponential
        and logarithmic maps (the ``calcW``/``calcWInv`` helpers).

        Both W and W^{-1} are of the form ``A * Omega + B * Omega^2 + C * I``.
        For small rotation angle ``theta``, some of the coefficients suffer
        from catastrophic cancellation when evaluated directly in terms of
        ``theta``, ``sin(theta)`` and ``cos(theta)``. The half-angle
        reformulations below are algebraically identical to the direct forms
        (verified in ``TestSimDetails`` below) but avoid the cancellation.
    """

    @staticmethod
    def W_A_theta_branch(theta):
        """ Coefficient A of W, sigma ~= 0 branch, direct form. """
        return (1 - sympy.cos(theta)) / theta**2

    @staticmethod
    def W_A_theta_branch_half_angle(theta):
        """ Half-angle reformulation of W's A coefficient. Since
            ``1 - cos(theta) == 2 * sin(theta / 2)**2`` exactly, this removes
            the cancellation entirely rather than merely widening the
            small-angle threshold around it. """
        half_theta = theta / 2
        return 2 * sympy.sin(half_theta)**2 / theta**2

    @staticmethod
    def W_inv_b_theta_branch(theta):
        """ Coefficient b of W^{-1}, sigma ~= 0 branch, direct form. """
        sin_theta = sympy.sin(theta)
        cos_theta = sympy.cos(theta)
        return (theta * sin_theta + 2 * cos_theta - 2) / \
            (2 * theta**2 * (cos_theta - 1))

    @staticmethod
    def W_inv_b_theta_branch_half_angle(theta):
        """ Half-angle reformulation of W^{-1}'s b coefficient. The direct
            form above has quartic-order cancellation in theta (both the
            numerator and, via cos(theta) - 1, the denominator vanish as
            theta**4); this reformulation reduces the remaining cancellation
            to cubic order, matching the conditioning of the analogous B
            coefficient of W and comfortably covered by the existing
            small-angle threshold. """
        half_theta = theta / 2
        sin_ht = sympy.sin(half_theta)
        cos_ht = sympy.cos(half_theta)
        return (sin_ht - half_theta * cos_ht) / (4 * half_theta**2 * sin_ht)

    @staticmethod
    def W_inv_denominator_direct(theta, sigma):
        """ scale^2 - 2*scale*cos(theta) + 1, the (shared, up to a factor
            of (scale - 1) for b) denominator of W^{-1}'s a and b
            coefficients in the branch where neither theta nor sigma is
            small on its own. """
        scale = sympy.exp(sigma)
        return scale**2 - 2 * scale * sympy.cos(theta) + 1

    @staticmethod
    def W_inv_denominator_half_angle(theta, sigma):
        """ Half-angle reformulation of the shared denominator above: a sum
            of two non-negative, individually well-conditioned squares
            (scaled by scale), removing the cancellation that occurs when
            theta and sigma are simultaneously close to zero (each
            individually large enough to bypass the small-value branches,
            but jointly small enough that the direct form above computes a
            near-zero result from an O(1) difference). """
        scale = sympy.exp(sigma)
        half_theta = theta / 2
        half_sigma = sigma / 2
        return 4 * scale * (sympy.sinh(half_sigma)**2 +
                            sympy.sin(half_theta)**2)

    @staticmethod
    def W_inv_a_generic_numerator_direct(theta, sigma):
        """ Numerator of W^{-1}'s a coefficient, branch where neither theta
            nor sigma is small, direct form. """
        scale = sympy.exp(sigma)
        s_sin_theta = scale * sympy.sin(theta)
        s_cos_theta = scale * sympy.cos(theta)
        return theta * s_cos_theta - theta - sigma * s_sin_theta

    @staticmethod
    def W_inv_a_generic_numerator_half_angle(theta, sigma):
        """ Half-angle reformulation of the a-coefficient numerator above,
            in terms of expm1(sigma) (represented here as scale - 1, sympy
            has no symbolic expm1) and half-angle sin/cos of theta and
            sigma. """
        scale = sympy.exp(sigma)
        half_theta = theta / 2
        half_sigma = sigma / 2
        sin_ht = sympy.sin(half_theta)
        cos_ht = sympy.cos(half_theta)
        expm1_sigma = scale - 1
        return (2 * half_theta * expm1_sigma * sympy.cos(theta) -
                4 * half_theta * sin_ht**2 -
                4 * half_sigma * scale * sin_ht * cos_ht)

    @staticmethod
    def W_inv_b_generic_numerator_direct(theta, sigma):
        """ Numerator of W^{-1}'s b coefficient (including the leading
            -scale factor), branch where neither theta nor sigma is small,
            direct form. """
        scale = sympy.exp(sigma)
        sin_theta = sympy.sin(theta)
        s_sin_theta = scale * sin_theta
        s_cos_theta = scale * sympy.cos(theta)
        return -scale * (theta * s_sin_theta - theta * sin_theta +
                         sigma * s_cos_theta - scale * sigma +
                         sigma * sympy.cos(theta) - sigma)

    @staticmethod
    def W_inv_b_generic_numerator_half_angle(theta, sigma):
        """ Half-angle reformulation of the b-coefficient numerator above,
            in terms of expm1(sigma) and half-angle sin(theta / 2). """
        scale = sympy.exp(sigma)
        half_theta = theta / 2
        sin_ht = sympy.sin(half_theta)
        expm1_sigma = scale - 1
        return -scale * (theta * sympy.sin(theta) * expm1_sigma -
                         2 * sigma * sin_ht**2 * (scale + 1))


class TestSimDetails(unittest.TestCase):
    def setUp(self):
        self.theta = sympy.symbols('theta', real=True)
        self.sigma = sympy.symbols('sigma', real=True)

    def test_W_A_half_angle_matches_direct_form(self):
        theta = self.theta
        direct = SimDetails.W_A_theta_branch(theta)
        half_angle = SimDetails.W_A_theta_branch_half_angle(theta)
        self.assertEqual(sympy.simplify(direct - half_angle), 0)
        self.assertEqual(sympy.limit(half_angle, theta, 0), sympy.Rational(1, 2))

    def test_W_inv_b_half_angle_matches_direct_form(self):
        theta = self.theta
        direct = SimDetails.W_inv_b_theta_branch(theta)
        half_angle = SimDetails.W_inv_b_theta_branch_half_angle(theta)
        self.assertEqual(sympy.simplify(direct - half_angle), 0)

        # Both forms must agree on the theta -> 0 Taylor limit used by the
        # small-angle branch in sim_details.hpp.
        series_direct = sympy.series(direct, theta, 0, 4).removeO()
        series_half_angle = sympy.series(half_angle, theta, 0, 4).removeO()
        self.assertEqual(sympy.simplify(series_direct - series_half_angle), 0)
        self.assertEqual(sympy.limit(half_angle, theta, 0), sympy.Rational(1, 12))

    @staticmethod
    def assert_identical(test_case, lhs, rhs):
        # sympy.simplify alone does not always collapse expressions mixing
        # exp/sin/cos/sinh; rewriting in terms of exp first reliably does.
        diff = sympy.simplify(sympy.expand((lhs - rhs).rewrite(sympy.exp)))
        test_case.assertEqual(diff, 0)

    def test_W_inv_denominator_half_angle_matches_direct_form(self):
        theta, sigma = self.theta, self.sigma
        direct = SimDetails.W_inv_denominator_direct(theta, sigma)
        half_angle = SimDetails.W_inv_denominator_half_angle(theta, sigma)
        self.assert_identical(self, direct, half_angle)

    def test_W_inv_a_generic_half_angle_matches_direct_form(self):
        theta, sigma = self.theta, self.sigma
        direct = SimDetails.W_inv_a_generic_numerator_direct(theta, sigma)
        half_angle = SimDetails.W_inv_a_generic_numerator_half_angle(
            theta, sigma)
        self.assert_identical(self, direct, half_angle)

    def test_W_inv_b_generic_half_angle_matches_direct_form(self):
        theta, sigma = self.theta, self.sigma
        direct = SimDetails.W_inv_b_generic_numerator_direct(theta, sigma)
        half_angle = SimDetails.W_inv_b_generic_numerator_half_angle(
            theta, sigma)
        self.assert_identical(self, direct, half_angle)


if __name__ == '__main__':
    unittest.main()
