// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

/// @file
/// Trigonometric functions of an angle with a removable singularity at zero,
/// evaluated without the cancellation of their closed forms.

#pragma once

#include <cmath>

#include "common.hpp"

namespace Sophus {
namespace details {

/**
 * Returns @f$(1 - \cos\theta) / \theta^2@f$.
 *
 * The half-angle identity @f$1 - \cos\theta = 2\sin^2(\theta/2)@f$ avoids
 * the cancellation of @f$1 - \cos\theta@f$ for small angles.
 */
template <class Scalar>
SOPHUS_FUNC Scalar oneMinusCosOverThetaSq(Scalar const& theta) {
  using std::sin;
  Scalar const theta_sq = theta * theta;
  if (theta_sq < Constants<Scalar>::epsilon()) {
    return Scalar(0.5) - theta_sq / Scalar(24);
  }
  Scalar const sin_half_theta_by_theta = sin(theta / Scalar(2)) / theta;
  return Scalar(2) * sin_half_theta_by_theta * sin_half_theta_by_theta;
}

/// Below this squared angle, the alternating series below converge to
/// machine precision, while closed forms such as
/// @f$(\theta - \sin\theta) / \theta^3@f$ lose up to a digit at its bound and
/// all digits towards zero.
constexpr double kSeriesThetaSqBound = 4;

/// The number of terms of the alternating series. For @f$\theta^2 < 4@f$, the
/// first omitted term of each series below is smaller than @f$10^{-19}@f$
/// times its sum.
constexpr int kSeriesTerms = 12;

/**
 * Returns the truncated series
 * @f[
 *   \sum_{k=0}^{n-1} (-\theta^2)^k \frac{w(k)}{(2k + m)!}
 * @f]
 * with @f$n@f$ = kSeriesTerms, evaluated by Horner's scheme.
 *
 * @param theta_sq The squared angle @f$\theta^2@f$.
 * @param m The order of the factorial of the first term.
 * @param weight The integer weight @f$w(k)@f$ of term @f$k@f$.
 */
template <class Scalar, class Weight>
SOPHUS_FUNC Scalar alternatingSeries(Scalar const& theta_sq, int m,
                                     Weight const& weight) {
  int const last_order = 2 * (kSeriesTerms - 1) + m;
  Scalar inverse_factorial(1);
  for (int i = 2; i <= last_order; ++i) {
    inverse_factorial /= Scalar(i);
  }
  Scalar sum(0);
  for (int k = kSeriesTerms - 1; k >= 0; --k) {
    sum = Scalar(weight(k)) * inverse_factorial - theta_sq * sum;
    // From 1/(2k + m)! to 1/(2k + m - 2)!.
    inverse_factorial *= Scalar((2 * k + m) * (2 * k + m - 1));
  }
  return sum;
}

}  // namespace details
}  // namespace Sophus
