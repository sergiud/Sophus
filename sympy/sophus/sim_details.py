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


class TestSimDetails(unittest.TestCase):
    def setUp(self):
        self.theta = sympy.symbols('theta', real=True)

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


if __name__ == '__main__':
    unittest.main()
