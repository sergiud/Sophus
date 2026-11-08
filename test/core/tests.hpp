// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>

#include <sophus/average.hpp>
#include <sophus/interpolate.hpp>
#include <sophus/num_diff.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/spline.hpp>

#include "matchers.hpp"
#include "registration.hpp"
#include "type_names.hpp"

#ifdef SOPHUS_CERES
#include <ceres/jet.h>
#endif

namespace Sophus {

template <class T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

/// The group elements, tangents and points the generic tests of a Lie group
/// run on. Each test of a Lie group specializes it with the static member
/// functions `groupElements`, `tangents` and `points`.
template <class LieGroup>
struct LieGroupTestData;

/// Applies `f` to each element of `values` and returns the results, so that
/// a property over all elements can be checked with a single container
/// matcher such as `Pointwise` or `Each`.
template <class Container, class Function>
auto mapped(Container const& values, Function f) {
  using Result = std::decay_t<decltype(f(*std::begin(values)))>;
  StdVector<Result> results;
  results.reserve(values.size());
  std::transform(std::begin(values), std::end(values),
                 std::back_inserter(results), f);
  return results;
}

/// Returns `count` matrices with entries drawn uniformly from [-1, 1].
template <class Matrix>
StdVector<Matrix> randomMatrices(int count) {
  StdVector<Matrix> matrices;
  std::generate_n(std::back_inserter(matrices), count,
                  [] { return Matrix(Matrix::Random()); });
  return matrices;
}

/// Scales an absolute tolerance by the largest magnitude in `reference`, but
/// not below 1, since the rounding error of a numerical derivative grows with
/// the magnitude of the derivative.
template <class Scalar, class Derived>
Scalar relativeTolerance(Scalar tolerance,
                         Eigen::MatrixBase<Derived> const& reference) {
  using std::max;
  return tolerance * max(Scalar(1), reference.cwiseAbs().maxCoeff());
}

template <typename Scalar>
Hyperplane<Scalar, 2> through(const Vector<Scalar, 2>* points) {
  return Hyperplane<Scalar, 2>::Through(points[0], points[1]);
}

template <typename Scalar>
Hyperplane<Scalar, 3> through(const Vector<Scalar, 3>* points) {
  return Hyperplane<Scalar, 3>::Through(points[0], points[1], points[2]);
}

/// Registers the generic tests of `LieGroup` with one test per entry of its
/// `LieGroupTestData`. The tests of a group named `SO3` with `float` scalars
/// form suites such as `SO3/LieGroupTest/float`, and each test is named
/// after the property and the indices of its data, such as
/// `ExpInvertsLog/3`.
template <class LieGroup>
class LieGroupTests {
 public:
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;
  using Point = typename LieGroup::Point;
  using HomogeneousPoint = typename LieGroup::HomogeneousPoint;
  using Line = typename LieGroup::Line;
  using Hyperplane = typename LieGroup::Hyperplane;
  using Adjoint = typename LieGroup::Adjoint;
  static int constexpr kDim = LieGroup::Dim;
  static int constexpr kN = LieGroup::N;
  static int constexpr kDoF = LieGroup::DoF;
  static int constexpr kNumParameters = LieGroup::num_parameters;

  /// Registers the properties every Lie group satisfies for any scalar type,
  /// including automatic differentiation types such as `ceres::Jet`.
  static bool registerGeneric(std::string const& group_name);

  /// Registers the properties that need a floating-point scalar type, such
  /// as numerical derivatives, matrix exponentials, interpolation, averages,
  /// random sampling and splines.
  static bool registerFloatingPoint(std::string const& group_name);

  /// Registers the properties of the left Jacobian, which only SO3 and SE3
  /// implement.
  static bool registerLeftJacobian(std::string const& group_name);

 private:
  static Scalar smallEps() { return Constants<Scalar>::epsilon(); }
  static Scalar smallEpsSqrt() { return Constants<Scalar>::epsilonSqrt(); }

  /// The rounding error of an identity between products of a few matrices,
  /// in units of the machine epsilon.
  static constexpr int kIdentityEpsilons = 64;
  /// The rounding error of interpolation, which composes logarithms,
  /// exponentials and products of elements with large translations, in
  /// units of the machine epsilon.
  static constexpr int kInterpolationEpsilons = 4096;
  /// The error of a central difference of a derivative in units of the
  /// square root of the machine epsilon.
  static constexpr int kNumDiffEpsilonSqrts = 10;

  static Scalar machineEps() { return std::numeric_limits<Scalar>::epsilon(); }
  static Scalar machineEpsSqrt() {
    using std::sqrt;
    return sqrt(machineEps());
  }
  static Scalar identityTolerance() {
    return Scalar(kIdentityEpsilons) * machineEps();
  }
  static Scalar interpolationTolerance() {
    return Scalar(kInterpolationEpsilons) * machineEps();
  }
  static Scalar numDiffTolerance() {
    return Scalar(kNumDiffEpsilonSqrts) * machineEpsSqrt();
  }

  /// The error of a central difference of a spline relative to the
  /// magnitude of its derivative, in units of the square root of the machine
  /// epsilon. The steps of the difference are bounded by the length of a
  /// segment, which raises the rounding error.
  static constexpr int kSplineNumDiffEpsilonSqrts = 64;

  template <class Derived>
  static Scalar splineNumDiffTolerance(
      Eigen::MatrixBase<Derived> const& reference) {
    return relativeTolerance(
        Scalar(kSplineNumDiffEpsilonSqrts) * machineEpsSqrt(), reference);
  }

  static StdVector<LieGroup> elements() {
    return LieGroupTestData<LieGroup>::groupElements();
  }
  static StdVector<Tangent> tangents() {
    return LieGroupTestData<LieGroup>::tangents();
  }
  static StdVector<Point> points() {
    return LieGroupTestData<LieGroup>::points();
  }

  static std::string scalarName() {
    return details::scalarName(details::TypeTag<LieGroup>{});
  }

  static std::string suiteName(std::string const& group_name,
                               char const* suite) {
    return group_name + "/" + suite + "/" + scalarName();
  }

  /// Joins indices of test data to the name of a test, such as `3_5`.
  template <class... Indices>
  static std::string indexName(Indices... indices) {
    std::string name;
    ((name += (name.empty() ? "" : "_") + std::to_string(indices)), ...);
    return name;
  }

  static void add(std::string const& suite, std::string const& name,
                  std::string const& value_param, int line,
                  std::function<void()> body) {
    registerTest({suite, name, scalarName(), value_param, __FILE__, line},
                 std::move(body));
  }

  static std::string describe(LieGroup const& g) {
    return ::testing::PrintToString(g.params());
  }

  template <class T>
  static std::string describe(T const& value) {
    return ::testing::PrintToString(value);
  }

  static Point map(Transformation const& T, Point const& p) {
    if constexpr (kDim == kN) {
      return T * p;
    } else {
      return T.template topLeftCorner<kN - 1, kN - 1>() * p +
             T.template topRightCorner<kN - 1, 1>();
    }
  }

  static bool hasShortestPathAmbiguity(LieGroup const& g) {
    return interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(g);
  }

  /// Control poses of a spline that interpolates between two elements. The
  /// intermediate poses are moved off the geodesic between the elements in
  /// different directions, so that the factors of a segment do not commute.
  static std::vector<LieGroup> splineControlPoses(LieGroup const& T_world_foo,
                                                  LieGroup const& T_world_bar) {
    int constexpr kNumIntermediatePoses = 4;
    int constexpr kOffsetDivisor = 10;
    StdVector<Tangent> const offsets = tangents();
    std::vector<LieGroup> control_poses;
    control_poses.push_back(interpolate(T_world_foo, T_world_bar, Scalar(0)));
    for (int i = 1; i <= kNumIntermediatePoses; ++i) {
      Scalar const p = Scalar(i) / Scalar(kNumIntermediatePoses + 1);
      Tangent const offset =
          offsets[static_cast<std::size_t>(i) % offsets.size()] /
          Scalar(kOffsetDivisor);
      control_poses.push_back(interpolate(T_world_foo, T_world_bar, p) *
                              LieGroup::exp(offset));
    }
    return control_poses;
  }

  static void adjointMatchesConjugationOfHat(LieGroup const& g) {
    Adjoint const Ad = g.Adj();
    Transformation const T = g.matrix();
    Transformation const T_inv = g.inverse().matrix();
    auto const ad1 =
        mapped(tangents(), [&Ad](Tangent const& x) { return Tangent(Ad * x); });
    auto const ad2 = mapped(tangents(), [&T, &T_inv](Tangent const& x) {
      return Tangent(LieGroup::vee(T * LieGroup::hat(x) * T_inv));
    });
    EXPECT_THAT(ad1, ::testing::Pointwise(IsNear(Scalar(10) * smallEps()), ad2))
        << "for each tangent";
  }

  static void copiesAndMapsPreserveTheElement(LieGroup foo_T_bar) {
    LieGroup const foo_T2_bar = foo_T_bar;
    EXPECT_THAT(foo_T2_bar, IsNear(foo_T_bar, smallEps()))
        << "copy constructor";

    LieGroup foo_T3_bar;
    foo_T3_bar = foo_T_bar;
    EXPECT_THAT(foo_T3_bar, IsNear(foo_T_bar, smallEps())) << "copy assignment";

    LieGroup const foo_T4_bar(foo_T_bar.matrix());
    EXPECT_THAT(foo_T4_bar, IsNear(foo_T_bar, smallEps()))
        << "constructor from homogeneous matrix";

    Eigen::Map<LieGroup> const foo_Tmap_bar(foo_T_bar.data());
    LieGroup const foo_T5_bar = foo_Tmap_bar;
    EXPECT_THAT(foo_T5_bar, IsNear(foo_T_bar, smallEps()))
        << "construction from Eigen::Map";

    Eigen::Map<LieGroup const> const foo_Tcmap_bar(foo_T_bar.data());
    LieGroup foo_T6_bar;
    foo_T6_bar = foo_Tcmap_bar;
    EXPECT_THAT(foo_T6_bar, IsNear(foo_T_bar, smallEps()))
        << "assignment from Eigen::Map of const";

    LieGroup identity;
    Eigen::Map<LieGroup> foo_Tmap2_bar(identity.data());
    foo_Tmap2_bar = foo_T_bar;
    EXPECT_THAT(identity, IsNear(foo_T_bar, smallEps()))
        << "assignment to Eigen::Map";
  }

  static void inPlaceProductMatchesProduct(LieGroup T1, LieGroup const& T2) {
    LieGroup const product = T1 * T2;
    T1 *= T2;
    EXPECT_THAT(T1, IsNear(product, smallEps()));
  }

  static void expInvertsLog(LieGroup const& T) {
    EXPECT_THAT(LieGroup::exp(T.log()), IsNear(T, smallEps()));
  }

  static void actsOnPointsLikeItsMatrix(LieGroup const& g) {
    using ::testing::Pointwise;
    Transformation const T = g.matrix();
    auto const expected =
        mapped(points(), [&T](Point const& p) { return map(T, p); });

    EXPECT_THAT(mapped(points(), [&g](Point const& p) { return Point(g * p); }),
                Pointwise(IsNear(smallEps()), expected))
        << "for points";
    EXPECT_THAT(mapped(points(),
                       [&g](Point const& p) {
                         HomogeneousPoint const hp = p.homogeneous();
                         HomogeneousPoint const hp_t = g * hp;
                         return Point(hp_t.hnormalized());
                       }),
                Pointwise(IsNear(smallEps()), expected))
        << "for homogeneous points";
    EXPECT_THAT(mapped(points(),
                       [&g](Point const& p) {
                         Eigen::Map<Point const> const p_map(p.data());
                         return Point(g * p_map);
                       }),
                Pointwise(IsNear(smallEps()), expected))
        << "for maps of points";
  }

  static void mapsLineThroughTransformedPoints(LieGroup const& g,
                                               Point const& p1,
                                               Point const& p2) {
    Line const l = Line::Through(p1, p2);
    Line const l_t = g * l;
    Point const p1_t = g * p1;
    Point const p2_t = g * p2;
    EXPECT_THAT(l_t.squaredDistance(p1_t), IsNear(Scalar(0), smallEps()));
    EXPECT_THAT(l_t.squaredDistance(p2_t), IsNear(Scalar(0), smallEps()));
    EXPECT_THAT(l_t.direction().squaredNorm(),
                IsNear(l.direction().squaredNorm(), smallEps()));
  }

  static void mapsPlaneThroughTransformedPoints(
      LieGroup const& g, StdVector<Point> const& plane_points) {
    std::size_t constexpr kPointDim = Point::RowsAtCompileTime;
    ASSERT_EQ(plane_points.size(), kPointDim);
    Point through_points[kPointDim];
    std::copy(plane_points.begin(), plane_points.end(), through_points);
    Hyperplane const plane = through(through_points);
    Hyperplane const plane_t = g * plane;

    EXPECT_THAT(mapped(plane_points,
                       [&g, &plane_t](Point const& p) {
                         Point const p_t = g * p;
                         return plane_t.signedDistance(p_t);
                       }),
                ::testing::Each(IsNear(Scalar(0), smallEps())))
        << "signed distances of the transformed points";
    EXPECT_THAT(plane_t.normal().squaredNorm(),
                IsNear(plane.normal().squaredNorm(), smallEps()));
  }

  static void lieBracketMatchesCommutatorOfHats(Tangent const& x) {
    Transformation const hat_x = LieGroup::hat(x);
    auto const brackets = mapped(tangents(), [&x](Tangent const& y) {
      return Tangent(LieGroup::lieBracket(x, y));
    });
    auto const commutators = mapped(tangents(), [&hat_x](Tangent const& y) {
      Transformation const hat_y = LieGroup::hat(y);
      return Tangent(LieGroup::vee(hat_x * hat_y - hat_y * hat_x));
    });
    EXPECT_THAT(brackets, ::testing::Pointwise(IsNear(smallEps()), commutators))
        << "for each second tangent";
  }

  static void veeInvertsHat(Tangent const& x) {
    EXPECT_THAT(Tangent(LieGroup::vee(LieGroup::hat(x))),
                IsNear(x, smallEps()));
  }

  static void heapAllocationIsAligned() {
    auto const element = std::make_unique<LieGroup>();
    EXPECT_EQ(
        reinterpret_cast<std::uintptr_t>(element.get()) % alignof(LieGroup),
        0u);
  }

  static void dxExpXMatchesNumericalDerivative(Tangent const& a) {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> const J = LieGroup::Dx_exp_x(a);
    Eigen::Matrix<Scalar, kNumParameters, kDoF> const J_num =
        vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
            [](Tangent const& x) -> Vector<Scalar, kNumParameters> {
              return LieGroup::exp(x).params();
            },
            a);
    EXPECT_THAT(J, IsNear(J_num, Scalar(3) * smallEpsSqrt()));
  }

  static void dxExpXAt0MatchesNumericalDerivative() {
    Tangent o;
    setToZero(o);
    Eigen::Matrix<Scalar, kNumParameters, kDoF> const J =
        LieGroup::Dx_exp_x_at_0();
    Eigen::Matrix<Scalar, kNumParameters, kDoF> const J_num =
        vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
            [](Tangent const& x) -> Vector<Scalar, kNumParameters> {
              return LieGroup::exp(x).params();
            },
            o);
    EXPECT_THAT(J, IsNear(J_num, smallEpsSqrt()));
  }

  static void dxThisMulExpXAt0MatchesNumericalDerivative(LieGroup const& T) {
    Tangent o;
    setToZero(o);
    Eigen::Matrix<Scalar, kNumParameters, kDoF> const J =
        T.Dx_this_mul_exp_x_at_0();
    Eigen::Matrix<Scalar, kNumParameters, kDoF> const J_num =
        vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
            [T](Tangent const& x) -> Vector<Scalar, kNumParameters> {
              return (T * LieGroup::exp(x)).params();
            },
            o);
    EXPECT_THAT(J, IsNear(J_num, numDiffTolerance()));
  }

  static void dxLogThisInvByXAtThisInvertsDxThisMulExpXAt0(LieGroup const& T) {
    Eigen::Matrix<Scalar, kDoF, kDoF> const J =
        T.Dx_log_this_inv_by_x_at_this() * T.Dx_this_mul_exp_x_at_0();
    EXPECT_THAT(J, IsNear(Eigen::Matrix<Scalar, kDoF, kDoF>::Identity(),
                          identityTolerance()));
  }

  static void dxLogThisInvByXAtThisMatchesNumericalDerivative(
      LieGroup const& T) {
    LieGroup const T_inv = T.inverse();
    // The parameters are perturbed in every direction, including those that
    // leave the group, such as the norm of a unit quaternion. The product
    // with T^{-1} defines the derivative in those directions, as Ceres
    // manifolds expect of their Minus operation.
    Eigen::Matrix<Scalar, kDoF, kNumParameters> const J_num =
        vectorFieldNumDiff<Scalar, kDoF, kNumParameters>(
            [T_inv](Vector<Scalar, kNumParameters> const& x)
                -> Vector<Scalar, kDoF> {
              return Vector<Scalar, kDoF>(
                  (T_inv * Eigen::Map<LieGroup const>(x.data())).log());
            },
            T.params());
    EXPECT_THAT(T.Dx_log_this_inv_by_x_at_this(),
                IsNear(J_num, numDiffTolerance()));
  }

  static void dxiExpXMatrixAt0MatchesNumericalDerivative(int i) {
    Transformation const Gi = LieGroup().Dxi_exp_x_matrix_at_0(i);
    Transformation const Gi2 = curveNumDiff(
        [i](Scalar xi) -> Transformation {
          Tangent x;
          setToZero(x);
          setElementAt(x, xi, i);
          return LieGroup::exp(x).matrix();
        },
        Scalar(0));
    EXPECT_THAT(Gi, IsNear(Gi2, smallEpsSqrt()));
  }

  static void dxExpXTimesPointAt0MatchesNumericalDerivative(
      Point const& point) {
    Matrix<Scalar, kDim, kDoF> const J =
        LieGroup::Dx_exp_x_times_point_at_0(point);
    Tangent t;
    setToZero(t);
    Matrix<Scalar, kDim, kDoF> const J_num =
        vectorFieldNumDiff<Scalar, kDim, kDoF>(
            [point](Tangent const& x) { return LieGroup::exp(x) * point; }, t);
    EXPECT_THAT(J, IsNear(J_num, smallEpsSqrt()));
  }

  static void expMatchesMatrixExponentialOfHat(Tangent const& omega) {
    Transformation const expmap_hat_x = LieGroup::hat(omega).exp();
    EXPECT_THAT(LieGroup::exp(omega),
                IsNear(expmap_hat_x, Scalar(10) * smallEps()));
  }

  static void interpolateHitsTheEndpoints(LieGroup const& foo_T_bar) {
    StdVector<LieGroup> const others = elements();
    EXPECT_THAT(mapped(others,
                       [&foo_T_bar](LieGroup const& foo_T_baz) {
                         return interpolate(foo_T_bar, foo_T_baz, Scalar(0));
                       }),
                ::testing::Each(IsNear(foo_T_bar, interpolationTolerance())))
        << "at alpha = 0 for each second element";
    EXPECT_THAT(mapped(others,
                       [&foo_T_bar](LieGroup const& foo_T_baz) {
                         return interpolate(foo_T_bar, foo_T_baz, Scalar(1));
                       }),
                ::testing::Pointwise(IsNear(interpolationTolerance()), others))
        << "at alpha = 1 for each second element";
  }

  // dash_T_foo * interp(foo_T_bar, foo_T_baz)
  // == interp(dash_T_foo * foo_T_bar, dash_T_foo * foo_T_baz)
  static void interpolateIsLeftInvariant(LieGroup const& foo_T_bar,
                                         LieGroup const& foo_T_baz,
                                         Scalar alpha) {
    LieGroup const foo_T_quiz = interpolate(foo_T_bar, foo_T_baz, alpha);
    StdVector<LieGroup> const dashes = elements();
    auto const actual = mapped(dashes, [&](LieGroup const& dash_T_foo) {
      return interpolate(LieGroup(dash_T_foo * foo_T_bar),
                         LieGroup(dash_T_foo * foo_T_baz), alpha);
    });
    auto const expected =
        mapped(dashes, [&foo_T_quiz](LieGroup const& dash_T_foo) {
          return LieGroup(dash_T_foo * foo_T_quiz);
        });
    EXPECT_THAT(actual, ::testing::Pointwise(IsNear(interpolationTolerance()),
                                             expected))
        << "for each element dash_T_foo";
  }

  // interp(foo_T_bar, foo_T_baz).inverse()
  // == interp(foo_T_bar.inverse(), foo_T_baz.inverse())
  static void interpolateIsInverseInvariant(LieGroup const& foo_T_bar,
                                            LieGroup const& foo_T_baz,
                                            Scalar alpha) {
    LieGroup const foo_T_quiz = interpolate(foo_T_bar, foo_T_baz, alpha);
    LieGroup const quiz_T_foo =
        interpolate(foo_T_bar.inverse(), foo_T_baz.inverse(), alpha);
    EXPECT_THAT(quiz_T_foo.inverse(),
                IsNear(foo_T_quiz, interpolationTolerance()));
  }

  // interp(bar_T_foo, baz_T_foo) * foo_T_dash
  // == interp(bar_T_foo * foo_T_dash, baz_T_foo * foo_T_dash)
  static void interpolateIsRightInvariant(LieGroup const& bar_T_foo,
                                          LieGroup const& baz_T_foo,
                                          Scalar alpha) {
    LieGroup const quiz_T_foo = interpolate(bar_T_foo, baz_T_foo, alpha);
    StdVector<LieGroup> const dashes = elements();
    auto const actual = mapped(dashes, [&](LieGroup const& foo_T_dash) {
      return interpolate(LieGroup(bar_T_foo * foo_T_dash),
                         LieGroup(baz_T_foo * foo_T_dash), alpha);
    });
    auto const expected =
        mapped(dashes, [&quiz_T_foo](LieGroup const& foo_T_dash) {
          return LieGroup(quiz_T_foo * foo_T_dash);
        });
    EXPECT_THAT(actual, ::testing::Pointwise(IsNear(interpolationTolerance()),
                                             expected))
        << "for each element foo_T_dash";
  }

  static void averageOfTwoIsTheirMidpoint(LieGroup const& foo_T_bar,
                                          LieGroup const& foo_T_baz) {
    using ::testing::Optional;
    int constexpr kMaxNumIterations = 20;
    LieGroup const foo_T_quiz = interpolate(foo_T_bar, foo_T_baz, Scalar(0.5));
    std::array<LieGroup, 2> const pair = {{foo_T_bar, foo_T_baz}};
    EXPECT_THAT(average(pair), Optional(IsNear(foo_T_quiz, smallEpsSqrt())));
    EXPECT_THAT(iterativeMean(pair, kMaxNumIterations),
                Optional(IsNear(foo_T_quiz, smallEpsSqrt())));
  }

  static void sampleUniformIsFinite() {
    int constexpr kNumSamples = 100;
    std::default_random_engine engine;
    StdVector<Vector<Scalar, kNumParameters>> samples;
    std::generate_n(std::back_inserter(samples), kNumSamples, [&engine] {
      return LieGroup::sampleUniform(engine).params();
    });
    EXPECT_THAT(samples, ::testing::Each(::testing::Each(IsFinite())));
  }

  static void splineSegmentsJoinContinuously(LieGroup const& T_world_foo,
                                             LieGroup const& T_world_bar) {
    BasisSplineImpl<LieGroup> const spline(
        splineControlPoses(T_world_foo, T_world_bar), Scalar(1));
    EXPECT_THAT(
        spline.parent_T_spline(0, Scalar(1)),
        IsNear(spline.parent_T_spline(1, Scalar(0)), interpolationTolerance()));
  }

  static void splineDerivativesMatchNumericalDerivatives(
      LieGroup const& T_world_foo, LieGroup const& T_world_bar) {
    Scalar const kU(0.5);
    BasisSplineImpl<LieGroup> const spline(
        splineControlPoses(T_world_foo, T_world_bar), Scalar(1));

    Transformation const Dt_parent_T_spline2 = curveNumDiff(
        [&spline](Scalar u_bar) -> Transformation {
          return spline.parent_T_spline(0, u_bar).matrix();
        },
        kU);
    EXPECT_THAT(spline.Dt_parent_T_spline(0, kU),
                IsNear(Dt_parent_T_spline2,
                       splineNumDiffTolerance(Dt_parent_T_spline2)));

    Transformation const Dt2_parent_T_spline2 = curveNumDiff(
        [&spline](Scalar u_bar) -> Transformation {
          return spline.Dt_parent_T_spline(0, u_bar);
        },
        kU);
    EXPECT_THAT(spline.Dt2_parent_T_spline(0, kU),
                IsNear(Dt2_parent_T_spline2,
                       splineNumDiffTolerance(Dt2_parent_T_spline2)));
  }

  static void basisSplineDerivativesMatchNumericalDerivatives(
      LieGroup const& T_world_foo, LieGroup const& T_world_bar, Scalar frac) {
    Scalar const kT0(1);
    Scalar const kDeltaT = Scalar(0.1);
    BasisSpline<LieGroup> const spline(
        splineControlPoses(T_world_foo, T_world_bar), kT0, kDeltaT);
    Scalar const t = kT0 + frac * kDeltaT;
    // Both points of the central difference must lie within the segment of
    // t: before t0 the spline is undefined, and across a knot its third
    // derivative jumps. For float, the default step size of sqrt(epsilon)
    // would exceed the distance to the knots for fractions near 0 and 1.
    Scalar const h = std::min({smallEpsSqrt(), frac * kDeltaT / Scalar(2),
                               (Scalar(1) - frac) * kDeltaT / Scalar(2)});

    Transformation const Dt_parent_T_spline2 = curveNumDiff(
        [&spline](Scalar t_bar) -> Transformation {
          return spline.parent_T_spline(t_bar).matrix();
        },
        t, h);
    EXPECT_THAT(spline.Dt_parent_T_spline(t),
                IsNear(Dt_parent_T_spline2,
                       splineNumDiffTolerance(Dt_parent_T_spline2)));

    Transformation const Dt2_parent_T_spline2 = curveNumDiff(
        [&spline](Scalar t_bar) -> Transformation {
          return spline.Dt_parent_T_spline(t_bar);
        },
        t, h);
    EXPECT_THAT(spline.Dt2_parent_T_spline(t),
                IsNear(Dt2_parent_T_spline2,
                       splineNumDiffTolerance(Dt2_parent_T_spline2)));
  }

  static void leftJacobianMatchesNumericalDerivative(Tangent const& x) {
    LieGroup const inv_exp_x = LieGroup::exp(x).inverse();
    // The derivative in the Lie group from first principles as a vector
    // field: D_x f(x) = D_h log(f(x + h) . f(x)^{-1})
    Matrix<Scalar, kDoF, kDoF> const J_num =
        vectorFieldNumDiff<Scalar, kDoF, kDoF>(
            [&inv_exp_x](Tangent const& x_plus_delta) {
              return (LieGroup::exp(x_plus_delta) * inv_exp_x).log();
            },
            x);
    EXPECT_THAT(LieGroup::leftJacobian(x), IsNear(J_num, numDiffTolerance()));
  }

  static void leftJacobianInverseIsItsInverse(Tangent const& x) {
    Matrix<Scalar, kDoF, kDoF> const J_inv = LieGroup::leftJacobianInverse(x);
    EXPECT_THAT(LieGroup::leftJacobian(x),
                IsNear(J_inv.inverse(), identityTolerance()));
  }
};

template <class LieGroup>
bool LieGroupTests<LieGroup>::registerGeneric(std::string const& group_name) {
  std::string const suite = suiteName(group_name, "LieGroupTest");
  StdVector<LieGroup> const g = elements();
  StdVector<Tangent> const x = tangents();
  StdVector<Point> const p = points();
  std::size_t constexpr kPointDim = Point::RowsAtCompileTime;

  for (std::size_t i = 0; i < g.size(); ++i) {
    std::string const value = "element " + describe(g[i]);
    add(suite, "AdjointMatchesConjugationOfHat/" + indexName(i), value,
        __LINE__, [e = g[i]] { adjointMatchesConjugationOfHat(e); });
    add(suite, "CopiesAndMapsPreserveTheElement/" + indexName(i), value,
        __LINE__, [e = g[i]] { copiesAndMapsPreserveTheElement(e); });
    add(suite, "ExpInvertsLog/" + indexName(i), value, __LINE__,
        [e = g[i]] { expInvertsLog(e); });
    add(suite, "ActsOnPointsLikeItsMatrix/" + indexName(i), value, __LINE__,
        [e = g[i]] { actsOnPointsLikeItsMatrix(e); });
    for (std::size_t j = 0; j + 1 < p.size(); ++j) {
      add(suite, "MapsLinesThroughTransformedPoints/" + indexName(i, j),
          value + ", points " + describe(p[j]) + " and " + describe(p[j + 1]),
          __LINE__, [e = g[i], p1 = p[j], p2 = p[j + 1]] {
            mapsLineThroughTransformedPoints(e, p1, p2);
          });
    }
    for (std::size_t j = 0; j + kPointDim <= p.size(); ++j) {
      auto const first = p.begin() + static_cast<std::ptrdiff_t>(j);
      StdVector<Point> const plane_points(first, first + kPointDim);
      add(suite, "MapsPlanesThroughTransformedPoints/" + indexName(i, j),
          value + ", points from #" + std::to_string(j), __LINE__,
          [e = g[i], plane_points] {
            mapsPlaneThroughTransformedPoints(e, plane_points);
          });
    }
  }
  for (std::size_t i = 0; i + 1 < g.size(); ++i) {
    add(suite, "InPlaceProductMatchesProduct/" + indexName(i),
        "elements " + describe(g[i]) + " and " + describe(g[i + 1]), __LINE__,
        [e1 = g[i], e2 = g[i + 1]] { inPlaceProductMatchesProduct(e1, e2); });
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    std::string const value = "tangent " + describe(x[i]);
    add(suite, "LieBracketMatchesCommutatorOfHats/" + indexName(i), value,
        __LINE__, [t = x[i]] { lieBracketMatchesCommutatorOfHats(t); });
    add(suite, "VeeInvertsHat/" + indexName(i), value, __LINE__,
        [t = x[i]] { veeInvertsHat(t); });
  }
  add(suite, "HeapAllocationIsAligned", "", __LINE__,
      [] { heapAllocationIsAligned(); });
  return true;
}

template <class LieGroup>
bool LieGroupTests<LieGroup>::registerFloatingPoint(
    std::string const& group_name) {
  std::string const suite = suiteName(group_name, "LieGroupFloatingPointTest");
  StdVector<LieGroup> const g = elements();
  StdVector<Tangent> const x = tangents();
  StdVector<Point> const p = points();
  std::array<Scalar, 4> const alphas = {Scalar(0.1), Scalar(0.5), Scalar(0.75),
                                        Scalar(0.99)};
  std::array<Scalar, 5> const fractions = {
      Scalar(0.01), Scalar(0.25), Scalar(0.5), Scalar(0.9), Scalar(0.99)};

  add(suite, "DxExpXAt0MatchesNumericalDerivative", "", __LINE__,
      [] { dxExpXAt0MatchesNumericalDerivative(); });
  add(suite, "SampleUniformIsFinite", "", __LINE__,
      [] { sampleUniformIsFinite(); });
  for (int i = 0; i < kDoF; ++i) {
    add(suite, "DxiExpXMatrixAt0MatchesNumericalDerivative/" + indexName(i),
        "generator " + std::to_string(i), __LINE__,
        [i] { dxiExpXMatrixAt0MatchesNumericalDerivative(i); });
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    std::string const value = "tangent " + describe(x[i]);
    add(suite, "DxExpXMatchesNumericalDerivative/" + indexName(i), value,
        __LINE__, [t = x[i]] { dxExpXMatchesNumericalDerivative(t); });
    add(suite, "ExpMatchesMatrixExponentialOfHat/" + indexName(i), value,
        __LINE__, [t = x[i]] { expMatchesMatrixExponentialOfHat(t); });
  }
  for (std::size_t i = 0; i < p.size(); ++i) {
    add(suite, "DxExpXTimesPointAt0MatchesNumericalDerivative/" + indexName(i),
        "point " + describe(p[i]), __LINE__,
        [q = p[i]] { dxExpXTimesPointAt0MatchesNumericalDerivative(q); });
  }

  for (std::size_t i = 0; i < g.size(); ++i) {
    std::string const value = "element " + describe(g[i]);
    add(suite, "DxThisMulExpXAt0MatchesNumericalDerivative/" + indexName(i),
        value, __LINE__,
        [e = g[i]] { dxThisMulExpXAt0MatchesNumericalDerivative(e); });
    add(suite, "DxLogThisInvByXAtThisInvertsDxThisMulExpXAt0/" + indexName(i),
        value, __LINE__,
        [e = g[i]] { dxLogThisInvByXAtThisInvertsDxThisMulExpXAt0(e); });
    add(suite,
        "DxLogThisInvByXAtThisMatchesNumericalDerivative/" + indexName(i),
        value, __LINE__,
        [e = g[i]] { dxLogThisInvByXAtThisMatchesNumericalDerivative(e); });
    add(suite, "InterpolateHitsTheEndpoints/" + indexName(i), value, __LINE__,
        [e = g[i]] { interpolateHitsTheEndpoints(e); });

    for (std::size_t j = 0; j < g.size(); ++j) {
      std::string const pair =
          "elements " + describe(g[i]) + " and " + describe(g[j]);
      add(suite, "SplineSegmentsJoinContinuously/" + indexName(i, j), pair,
          __LINE__,
          [a = g[i], b = g[j]] { splineSegmentsJoinContinuously(a, b); });
      add(suite,
          "SplineDerivativesMatchNumericalDerivatives/" + indexName(i, j), pair,
          __LINE__, [a = g[i], b = g[j]] {
            splineDerivativesMatchNumericalDerivatives(a, b);
          });
      for (std::size_t k = 0; k < fractions.size(); ++k) {
        add(suite,
            "BasisSplineDerivativesMatchNumericalDerivatives/" +
                indexName(i, j, k),
            pair + ", fraction " + describe(fractions[k]), __LINE__,
            [a = g[i], b = g[j], frac = fractions[k]] {
              basisSplineDerivativesMatchNumericalDerivatives(a, b, frac);
            });
      }

      // Interpolation is ambiguous where two shortest paths exist, such as
      // for rotations by pi, so these pairs have no tests.
      bool const left_ambiguous =
          hasShortestPathAmbiguity(g[i].inverse() * g[j]);
      bool const right_ambiguous =
          hasShortestPathAmbiguity(g[i] * g[j].inverse());
      if (!left_ambiguous) {
        add(suite, "AverageOfTwoIsTheirMidpoint/" + indexName(i, j), pair,
            __LINE__,
            [a = g[i], b = g[j]] { averageOfTwoIsTheirMidpoint(a, b); });
      }
      for (std::size_t k = 0; k < alphas.size(); ++k) {
        std::string const value_k = pair + ", alpha " + describe(alphas[k]);
        if (!left_ambiguous) {
          add(suite, "InterpolateIsLeftInvariant/" + indexName(i, j, k),
              value_k, __LINE__, [a = g[i], b = g[j], alpha = alphas[k]] {
                interpolateIsLeftInvariant(a, b, alpha);
              });
          add(suite, "InterpolateIsInverseInvariant/" + indexName(i, j, k),
              value_k, __LINE__, [a = g[i], b = g[j], alpha = alphas[k]] {
                interpolateIsInverseInvariant(a, b, alpha);
              });
        }
        if (!right_ambiguous) {
          add(suite, "InterpolateIsRightInvariant/" + indexName(i, j, k),
              value_k, __LINE__, [a = g[i], b = g[j], alpha = alphas[k]] {
                interpolateIsRightInvariant(a, b, alpha);
              });
        }
      }
    }
  }
  return true;
}

template <class LieGroup>
bool LieGroupTests<LieGroup>::registerLeftJacobian(
    std::string const& group_name) {
  std::string const suite = suiteName(group_name, "LieGroupLeftJacobianTest");
  StdVector<Tangent> const x = tangents();
  for (std::size_t i = 0; i < x.size(); ++i) {
    std::string const value = "tangent " + describe(x[i]);
    add(suite, "LeftJacobianMatchesNumericalDerivative/" + indexName(i), value,
        __LINE__, [t = x[i]] { leftJacobianMatchesNumericalDerivative(t); });
    add(suite, "LeftJacobianInverseIsItsInverse/" + indexName(i), value,
        __LINE__, [t = x[i]] { leftJacobianInverseIsItsInverse(t); });
  }
  return true;
}

/// Registers the generic tests of `LieGroup`, and the floating-point tests if
/// its scalar type is a floating-point type.
template <class LieGroup>
void registerLieGroupTestsOf(std::string const& group_name) {
  LieGroupTests<LieGroup>::registerGeneric(group_name);
  if constexpr (std::is_floating_point_v<typename LieGroup::Scalar>) {
    LieGroupTests<LieGroup>::registerFloatingPoint(group_name);
  }
}

/// Registers the tests of each Lie group type in `LieGroups`.
template <class... LieGroups>
bool registerLieGroupTests(std::string const& group_name) {
  (registerLieGroupTestsOf<LieGroups>(group_name), ...);
  return true;
}

template <class Scalar>
StdVector<SE3<Scalar>> getTestSE3s() {
  Scalar const kPi = Constants<Scalar>::pi();
  StdVector<SE3<Scalar>> se3_vec;
  se3_vec.push_back(SE3<Scalar>(
      SO3<Scalar>::exp(Vector3<Scalar>(Scalar(0.2), Scalar(0.5), Scalar(0.0))),
      Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(0))));
  se3_vec.push_back(SE3<Scalar>(
      SO3<Scalar>::exp(Vector3<Scalar>(Scalar(0.2), Scalar(0.5), Scalar(-1.0))),
      Vector3<Scalar>(Scalar(10), Scalar(0), Scalar(0))));
  se3_vec.push_back(
      SE3<Scalar>::trans(Vector3<Scalar>(Scalar(0), Scalar(100), Scalar(5))));
  se3_vec.push_back(SE3<Scalar>::rotZ(Scalar(0.00001)));
  se3_vec.push_back(
      SE3<Scalar>::trans(Scalar(0), Scalar(-0.00000001), Scalar(0.0000000001)) *
      SE3<Scalar>::rotZ(Scalar(0.00001)));
  se3_vec.push_back(SE3<Scalar>::transX(Scalar(0.01)) *
                    SE3<Scalar>::rotZ(Scalar(0.00001)));
  se3_vec.push_back(SE3<Scalar>::trans(Scalar(4), Scalar(-5), Scalar(0)) *
                    SE3<Scalar>::rotX(kPi));
  se3_vec.push_back(
      SE3<Scalar>(SO3<Scalar>::exp(
                      Vector3<Scalar>(Scalar(0.2), Scalar(0.5), Scalar(0.0))),
                  Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(0))) *
      SE3<Scalar>::rotX(kPi) *
      SE3<Scalar>(SO3<Scalar>::exp(Vector3<Scalar>(Scalar(-0.2), Scalar(-0.5),
                                                   Scalar(-0.0))),
                  Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(0))));
  se3_vec.push_back(
      SE3<Scalar>(SO3<Scalar>::exp(
                      Vector3<Scalar>(Scalar(0.3), Scalar(0.5), Scalar(0.1))),
                  Vector3<Scalar>(Scalar(2), Scalar(0), Scalar(-7))) *
      SE3<Scalar>::rotX(kPi) *
      SE3<Scalar>(SO3<Scalar>::exp(Vector3<Scalar>(Scalar(-0.3), Scalar(-0.5),
                                                   Scalar(-0.1))),
                  Vector3<Scalar>(Scalar(0), Scalar(6), Scalar(0))));
  return se3_vec;
}

template <class T>
StdVector<SE2<T>> getTestSE2s() {
  T const kPi = Constants<T>::pi();
  StdVector<SE2<T>> se2_vec;
  se2_vec.push_back(SE2<T>());
  se2_vec.push_back(SE2<T>(SO2<T>(T(0.2)), Vector2<T>(10, 0)));
  se2_vec.push_back(SE2<T>::transY(100));
  se2_vec.push_back(SE2<T>::trans(Vector2<T>(1, 2)));
  se2_vec.push_back(SE2<T>(SO2<T>(-1.), Vector2<T>(20, -1)));
  se2_vec.push_back(
      SE2<T>(SO2<T>(T(0.00001)), Vector2<T>(-0.00000001, 0.0000000001)));
  se2_vec.push_back(SE2<T>(SO2<T>(T(0.3)), Vector2<T>(2, 0)) *
                    SE2<T>::rot(kPi) *
                    SE2<T>(SO2<T>(T(-0.3)), Vector2<T>(0, 6)));
  return se2_vec;
}

/// Returns the raw parameters behind `data()` of a group element or map.
template <class Group>
auto rawParams(Group const& g) {
  using Scalar = typename Group::Scalar;
  return Eigen::Map<Vector<Scalar, Group::num_parameters> const>(g.data());
}

/// The scalar types every test runs for.
using FloatingPointScalars = ::testing::Types<double, float>;

}  // namespace Sophus
