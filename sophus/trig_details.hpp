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

}  // namespace details
}  // namespace Sophus
