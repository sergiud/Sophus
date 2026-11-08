// Copyright 2022 Sergiu Deitsch <sergiu.deitsch@gmail.com>

/// @file
/// Numerical differentiation using finite differences

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <type_traits>
#include <utility>

#include "types.hpp"

namespace Sophus {

namespace details {
template <class Scalar>
class Curve {
 public:
  template <class Fn>
  static auto num_diff(Fn curve, Scalar t, Scalar h) -> decltype(curve(t)) {
    using ReturnType = decltype(curve(t));
    static_assert(std::is_floating_point_v<Scalar>,
                  "Scalar must be a floating point type.");
    static_assert(IsFloatingPoint<ReturnType>::value,
                  "ReturnType must be either a floating point scalar, "
                  "vector or matrix.");

    // The step grows with the magnitude of the argument, so that it stays
    // above the spacing of the floating-point numbers there.
    using std::abs;
    using std::max;
    Scalar const step = h * max(Scalar(1), abs(t));
    return (curve(t + step) - curve(t - step)) / (Scalar(2) * step);
  }
};

template <class Scalar, int N, int M>
class VectorField {
 public:
  static Eigen::Matrix<Scalar, N, M> num_diff(
      std::function<Sophus::Vector<Scalar, N>(Sophus::Vector<Scalar, M>)>
          vector_field,
      Sophus::Vector<Scalar, M> const& a, Scalar eps) {
    static_assert(std::is_floating_point_v<Scalar>,
                  "Scalar must be a floating point type.");
    Eigen::Matrix<Scalar, N, M> J;
    for (int i = 0; i < M; ++i) {
      // See Curve::num_diff for the choice of the step.
      using std::abs;
      using std::max;
      Sophus::Vector<Scalar, M> forward = a;
      Sophus::Vector<Scalar, M> backward = a;
      Scalar const step = eps * max(Scalar(1), abs(a[i]));
      forward[i] += step;
      backward[i] -= step;
      J.col(i) =
          (vector_field(forward) - vector_field(backward)) / (Scalar(2) * step);
    }

    return J;
  }
};

template <class Scalar, int N>
class VectorField<Scalar, N, 1> {
 public:
  static Eigen::Matrix<Scalar, N, 1> num_diff(
      std::function<Sophus::Vector<Scalar, N>(Scalar)> vector_field,
      Scalar const& a, Scalar eps) {
    return details::Curve<Scalar>::num_diff(std::move(vector_field), a, eps);
  }
};
}  // namespace details

/// Calculates the derivative of a curve at a point ``t``.
///
/// Here, a curve is a function from a Scalar to a Euclidean space. Thus, it
/// returns either a Scalar, a vector or a matrix. The central difference
/// steps by ``h * max(1, |t|)``.
///
template <class Scalar, class Fn>
auto curveNumDiff(Fn curve, Scalar t,
                  Scalar h = Constants<Scalar>::epsilonSqrt())
    -> decltype(details::Curve<Scalar>::num_diff(std::move(curve), t, h)) {
  return details::Curve<Scalar>::num_diff(std::move(curve), t, h);
}

/// Calculates the derivative of a vector field at a point ``a``.
///
/// Here, a vector field is a function from a vector space to another vector
/// space. The central difference steps each coordinate ``a_i`` by
/// ``eps * max(1, |a_i|)``.
///
template <class Scalar, int N, int M, class ScalarOrVector, class Fn>
Eigen::Matrix<Scalar, N, M> vectorFieldNumDiff(
    Fn vector_field, ScalarOrVector const& a,
    Scalar eps = Constants<Scalar>::epsilonSqrt()) {
  return details::VectorField<Scalar, N, M>::num_diff(std::move(vector_field),
                                                      a, eps);
}

}  // namespace Sophus
