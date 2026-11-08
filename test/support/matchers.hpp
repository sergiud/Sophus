// SPDX-FileCopyrightText: 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <Eigen/Core>

#include <sophus/cartesian.hpp>
#include <sophus/rotation_matrix.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim2.hpp>
#include <sophus/sim3.hpp>
#include <sophus/types.hpp>

namespace Eigen {

/// Prints vectors on one line and matrices row by row on their own lines, so
/// that failure messages keep the layout of a matrix.
template <class Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
void PrintTo(Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> const& m,
             std::ostream* os) {
  if (m.rows() == 1 || m.cols() == 1) {
    IOFormat const kVectorFormat(FullPrecision, DontAlignCols, ", ", ", ", "",
                                 "", "[", "]");
    *os << m.reshaped().transpose().format(kVectorFormat);
  } else {
    IOFormat const kMatrixFormat(FullPrecision, DontAlignCols, ", ", ",\n ",
                                 "[", "]", "[", "]");
    *os << '\n' << m.format(kMatrixFormat);
  }
}

}  // namespace Eigen

namespace Sophus {
namespace details {

template <class T, class = void>
struct IsLieGroup : std::false_type {};

template <class T>
struct IsLieGroup<T, std::void_t<typename T::Transformation,
                                 decltype(std::declval<T const&>().matrix()),
                                 decltype(std::declval<T const&>().params())>>
    : std::true_type {};

template <class T, class = void>
struct IsEigenObject : std::false_type {};

template <class T>
struct IsEigenObject<T, std::void_t<typename T::PlainObject>> : std::true_type {
};

/// Converts a value to a type `maxMetric` accepts: Eigen expressions to plain
/// matrices, Lie group elements to their matrices and scalars to themselves.
template <class T>
auto comparable(T const& value) {
  if constexpr (IsLieGroup<T>::value) {
    return value.matrix().eval();
  } else if constexpr (IsEigenObject<T>::value) {
    return value.eval();
  } else {
    return value;
  }
}

template <class T>
using Comparable = decltype(comparable(std::declval<T const&>()));

template <class Group>
void printLieGroup(Group const& group, std::ostream* os) {
  Eigen::PrintTo(group.matrix().eval(), os);
}

}  // namespace details

// Google Test prints Lie group elements as their matrices instead of their
// bytes. Each group needs its own overload to be more specialized than the
// generic PrintTo of Google Test.
template <class Scalar, int M, int Options>
void PrintTo(Cartesian<Scalar, M, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

template <class Scalar, int Options>
void PrintTo(SO2<Scalar, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

template <class Scalar, int Options>
void PrintTo(SO3<Scalar, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

template <class Scalar, int Options>
void PrintTo(SE2<Scalar, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

template <class Scalar, int Options>
void PrintTo(SE3<Scalar, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

template <class Scalar, int Options>
void PrintTo(RxSO2<Scalar, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

template <class Scalar, int Options>
void PrintTo(RxSO3<Scalar, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

template <class Scalar, int Options>
void PrintTo(Sim2<Scalar, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

template <class Scalar, int Options>
void PrintTo(Sim3<Scalar, Options> const& g, std::ostream* os) {
  details::printLieGroup(g, os);
}

namespace details {

MATCHER_P2(IsNearMatcher, expected, tolerance,
           std::string(negation ? "isn't" : "is") +
               " within max-norm distance " +
               ::testing::PrintToString(tolerance) + " of " +
               ::testing::PrintToString(expected)) {
  auto const distance = maxMetric(expected_type(comparable(arg)), expected);
  *result_listener << "whose max-norm distance is "
                   << ::testing::PrintToString(distance);
  // Written such that a NaN distance does not match.
  return distance <= tolerance;
}

}  // namespace details

/// Matches a scalar, an Eigen matrix or a Lie group element whose largest
/// absolute difference to `expected` is at most `tolerance`, the metric of
/// `Sophus::maxMetric`. Lie group elements are compared by their matrices.
template <class Expected, class Tolerance>
auto IsNear(Expected const& expected, Tolerance tolerance) {
  // Evaluating expressions here avoids holding references to temporaries.
  return details::IsNearMatcher(details::comparable(expected), tolerance);
}

namespace details {

MATCHER_P(IsNearPairMatcher, tolerance,
          std::string(negation ? "aren't" : "are") +
              " within max-norm distance " +
              ::testing::PrintToString(tolerance) + " of each other") {
  return ::testing::ExplainMatchResult(IsNear(std::get<1>(arg), tolerance),
                                       std::get<0>(arg), result_listener);
}

}  // namespace details

/// Matches a pair of values `(actual, expected)` such as `Pointwise` forms,
/// where `actual` is near `expected` as `IsNear(expected, tolerance)` defines.
template <class Tolerance>
auto IsNear(Tolerance tolerance) {
  return details::IsNearPairMatcher(tolerance);
}

/// Matches a quaternion that represents the same rotation as `expected` up
/// to `tolerance`. Since q and -q represent the same rotation, the distance
/// is the smaller max-norm distance of the coefficients to those of
/// `expected` and of its negation.
MATCHER_P2(IsNearQuaternion, expected, tolerance,
           std::string(negation ? "isn't" : "is") +
               " within max-norm distance " +
               ::testing::PrintToString(tolerance) + " of " +
               ::testing::PrintToString(expected) +
               " or of its negation, ignoring the sign of the quaternion") {
  using std::min;
  auto const coeffs = arg.coeffs().eval();
  auto const expected_coeffs = expected.coeffs().eval();
  auto const distance =
      min(maxMetric(coeffs, expected_coeffs),
          maxMetric(coeffs, decltype(expected_coeffs)(-expected_coeffs)));
  *result_listener << "whose max-norm distance up to sign is "
                   << ::testing::PrintToString(distance);
  return distance <= tolerance;
}

/// Matches a scalar that is neither infinite nor NaN.
MATCHER(IsFinite, negation ? "isn't finite" : "is finite") {
  using std::isfinite;
  return isfinite(arg);
}

/// Matches a scalar or a vector whose squared Euclidean norm matches
/// `matcher`.
MATCHER_P(HasSquaredNorm, matcher,
          "has a squared norm that " +
              ::testing::DescribeMatcher<double>(matcher, negation)) {
  auto const squared_norm = squaredNorm(details::comparable(arg));
  *result_listener << "whose squared norm is "
                   << ::testing::PrintToString(squared_norm);
  return ::testing::Matches(matcher)(squared_norm);
}

/// Matches a square matrix that is a positive multiple of a rotation matrix,
/// see `Sophus::isScaledOrthogonalAndPositive`.
MATCHER(IsScaledOrthogonalAndPositive,
        negation ? "isn't a positive multiple of a rotation matrix"
                 : "is a positive multiple of a rotation matrix") {
  *result_listener << "whose determinant is "
                   << ::testing::PrintToString(arg.determinant());
  return isScaledOrthogonalAndPositive(arg);
}

}  // namespace Sophus
